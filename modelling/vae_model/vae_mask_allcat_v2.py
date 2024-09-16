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

# concat all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)


# split training, validation, and testing (80% & 20%)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)


### model
import tensorflow as tf
import matplotlib.pyplot as plt


class RandomMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(RandomMaskingLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        # input shape
        batch_size = tf.shape(inputs)[0]

        # randomly select one of the four sections to mask (i.e. DNAm, K9, K27, expression)
        mask_index = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

        # create a mask for each section: tf.concat() used to break input into masked and non-masked segments then concat them back together
        masks = [
            tf.concat(
                [
                    tf.fill([batch_size, 4000], -1.0),
                    tf.zeros([batch_size, 8001], dtype=tf.float32),
                ],
                axis=1,
            ),  # DNAm
            tf.concat(
                [
                    tf.zeros([batch_size, 4000], dtype=tf.float32),
                    tf.fill([batch_size, 4000], -1.0),
                    tf.zeros([batch_size, 4001], dtype=tf.float32),
                ],
                axis=1,
            ),  # K9
            tf.concat(
                [
                    tf.zeros([batch_size, 8000], dtype=tf.float32),
                    tf.fill([batch_size, 4000], -1.0),
                    tf.zeros([batch_size, 1], dtype=tf.float32),
                ],
                axis=1,
            ),  # K27
            tf.concat(
                [
                    tf.zeros([batch_size, 12000], dtype=tf.float32),
                    tf.fill([batch_size, 1], -1.0),
                ],
                axis=1,
            ),  # expression
        ]

        # select mask for the current batch
        mask = tf.gather(masks, mask_index)

        # apply mask to the input data
        masked_inputs = tf.where(mask != 0, mask, inputs)

        return masked_inputs, mask, mask_index


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


class CustomLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, inputs, reconstructions, mask):
        # Masking logic
        mask_indices = tf.cast(mask == -1, tf.float32)

        # Clip predictions to avoid log(0) or log(negative number) issues
        reconstructions_clipped = tf.clip_by_value(
            reconstructions, clip_value_min=1e-7, clip_value_max=1 - 1e-7
        )

        # Compute binary cross-entropy loss without creating an instance - clipping errors to avoid exploding gradient
        loss = tf.keras.backend.binary_crossentropy(inputs, reconstructions_clipped)

        loss = tf.reshape(
            loss, tf.shape(inputs)
        )  # Ensure loss shape is [batch_size, 12001]

        # Ensure that the loss tensor has the shape [batch_size, 12001]
        masked_loss = tf.reduce_sum(loss * mask_indices, axis=1) / tf.reduce_sum(
            mask_indices, axis=1
        )

        # Avoid division by zero
        masked_loss = tf.where(
            tf.reduce_sum(mask_indices, axis=1) == 0, 0.0, masked_loss
        )

        return tf.reduce_mean(masked_loss)


### Model class (incorporates the above masking and loss functions)
class VAEWithCustomLoss(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEWithCustomLoss, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss_layer = CustomLossLayer()

    def call(self, inputs, training=None):
        tf.print(
            "\nStarting batch processing..."
        )  # print to better understand the masking process (i.e. which situations cause spikes in error)
        masked_inputs, mask, mask_index = RandomMaskingLayer()(inputs)
        codings_mean, codings_log_var, codings = self.encoder(masked_inputs)
        reconstructions = self.decoder(codings)

        loss = self.custom_loss_layer(inputs, reconstructions, mask)
        if training:
            self.add_loss(loss)
        tf.print(" - Selected Subset:", mask_index)
        tf.print("Finished batch processing...")
        return reconstructions


# Define the encoder
codings_size = 50
inputs = tf.keras.layers.Input(shape=[12001])
masked_inputs, mask, mask_index = RandomMaskingLayer()(inputs)
x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(
    masked_inputs  # use he_normal to avoid exploding gradient
)
x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
codings_mean = tf.keras.layers.Dense(codings_size)(x)  # mean
codings_log_var = tf.keras.layers.Dense(codings_size)(x)  # log_var
codings = Sampling()([codings_mean, codings_log_var])
codings_mean, codings_log_var = KLDivergenceLayer()([codings_mean, codings_log_var])
variational_encoder = tf.keras.Model(
    inputs=[inputs], outputs=[codings_mean, codings_log_var, codings]
)

# Define the decoder
decoder_inputs = tf.keras.layers.Input(shape=[codings_size])
x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(
    decoder_inputs
)
x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(x)
x = tf.keras.layers.Dense(12001, activation="sigmoid")(x)  # Sigmoid for binary
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[x])

# Instantiate and compile the VAE model
vae_model = VAEWithCustomLoss(encoder=variational_encoder, decoder=variational_decoder)
vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

# Train the model
history = vae_model.fit(
    X_train,
    X_train,
    epochs=150,
    batch_size=128,
    validation_data=(X_val, X_val),
    verbose=1,
)

# Plot training and validation loss
# Create a figure with 2 subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Regular loss plot
axs[0].plot(history.history["loss"], label="Training Loss")
axs[0].plot(history.history["val_loss"], label="Validation Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].set_title("Training and Validation Loss")

# Plot 2: Loss plot with log y-axis
axs[1].plot(history.history["loss"], label="Training Loss")
axs[1].plot(history.history["val_loss"], label="Validation Loss")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss (Log Scale)")
axs[1].set_yscale("log")
axs[1].legend()
axs[1].set_title("Training and Validation Loss (Log Scale)")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig("vae_mask_allcat_v2.png")
