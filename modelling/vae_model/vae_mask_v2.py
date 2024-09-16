import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


### DATA PREPARATION
gene_matrix_array = np.load("gene_matrix_list.npy")[:200]
rna_expression_df = pd.read_csv("rna_expression_list.csv").iloc[:200]

# Check order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# Separate modification types
dnam_features = gene_matrix_array[:, :, 0]
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]
rna_expression = (rna_expression_df["expression"].values > 0).astype(
    int
)  # convert expression to binary
expression_values = rna_expression.reshape(-1, 1)  # Shape (58780, 1)

# Concatenate all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)


# Split into training, validation, and testing (70% training, 10% validation, 20% testing)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)


### Masking logic / layer
class RandomMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomMaskingLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        # Shape of the input
        batch_size = tf.shape(inputs)[0]

        # Randomly select one of the four sections to mask
        mask_index = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

        # Create masks for each section: tf.concat() used to break input into masked and non-masked segments then concat them back together
        masks = [
            tf.concat(
                [tf.fill([batch_size, 4000], -1), tf.zeros([batch_size, 8001])], axis=1
            ),  # DNAm
            tf.concat(
                [
                    tf.zeros([batch_size, 4000]),
                    tf.fill([batch_size, 4000], -1),
                    tf.zeros([batch_size, 4001]),
                ],
                axis=1,
            ),  # K9
            tf.concat(
                [
                    tf.zeros([batch_size, 8000]),
                    tf.fill([batch_size, 4000], -1),
                    tf.zeros([batch_size, 1]),
                ],
                axis=1,
            ),  # K27
            tf.concat(
                [tf.zeros([batch_size, 12000]), tf.fill([batch_size, 1], -1)], axis=1
            ),  # expression
        ]

        # Select the mask for the current batch
        mask = tf.gather(masks, mask_index)

        # Apply the mask to the input data
        masked_inputs = tf.where(mask != 0, mask, inputs)

        return masked_inputs, mask


### VAE
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


class KLDivergenceLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        codings_mean, codings_log_var = inputs
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
            axis=-1,
        )
        self.add_loss(tf.reduce_mean(kl_loss) / 12001.0)
        return inputs


### Custome loss function to track loss for only the masked regions
def masked_loss_function(inputs, reconstructions, mask):
    # binary crossentropy for only masked section
    mask_indices = tf.cast(
        mask == -1, tf.float32
    )  # get indices for masked values to filter for loss calc
    loss = tf.keras.losses.binary_crossentropy(
        inputs, reconstructions
    )  # calc BCE across all input vector and reconstruction vector
    masked_loss = tf.reduce_sum(loss * mask_indices, axis=1) / tf.reduce_sum(
        mask_indices, axis=1
    )  # filter out loss of non-masked regions across batches and normalise (by dividing by number of elements)
    return tf.reduce_mean(masked_loss)  # take the mean


codings_size = 50

# ENCODER
inputs = tf.keras.layers.Input(shape=[12001])
masked_inputs, mask = RandomMaskingLayer()(inputs)
x = tf.keras.layers.Dense(512, activation="relu")(masked_inputs)
x = tf.keras.layers.Dense(256, activation="relu")(x)
codings_mean = tf.keras.layers.Dense(codings_size)(x)  # mean
codings_log_var = tf.keras.layers.Dense(codings_size)(x)  # log_var
codings = Sampling()([codings_mean, codings_log_var])

# KL Divergence Layer
codings_mean, codings_log_var = KLDivergenceLayer()([codings_mean, codings_log_var])

# define VAE
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)

# DECODER
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(256, activation="relu")(decoder_inputs)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dense(12001, activation="sigmoid")(x)  # Sigmoid for binary

# output match original input
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[x])
_, _, codings = variational_encoder(inputs)  # generate codings
reconstructions = variational_decoder(codings)  # decode codings

# Define the VAE model
variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

# compile
variational_ae.add_loss(masked_loss_function(inputs, reconstructions, mask))
variational_ae.compile(optimizer="adam")

# train using train and valid set
history = variational_ae.fit(
    X_train,
    X_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_val, X_val),
    verbose=1,
)

import matplotlib.pyplot as plt

# plot training and val loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("vae_v1.png")
