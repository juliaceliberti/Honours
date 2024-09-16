import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


### DATA PREPARATION
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

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
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.66, random_state=42)


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


codings_size = 300

# ENCODER
inputs = tf.keras.layers.Input(shape=[12001])
x = tf.keras.layers.Dense(512, activation="relu")(inputs)
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

# compile (reconstruction loss added here)
variational_ae.compile(loss="binary_crossentropy", optimizer="adam")

# Adding early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

# train using train and valid set
history = variational_ae.fit(
    X_train,
    X_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
)

import matplotlib.pyplot as plt

# plot training and val loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("vae_v1.png")
