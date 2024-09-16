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
import tensorflow as tf
import matplotlib.pyplot as plt


class FixedMaskingLayer(
    tf.keras.layers.Layer
):  # masks a single subset of the input data (passed into the VAE class)
    def __init__(self, mask_index, **kwargs):
        super(FixedMaskingLayer, self).__init__(**kwargs)
        self.mask_index = mask_index  # Fixed subset to mask

    def call(self, inputs, training=None):
        # Shape of the input
        batch_size = tf.shape(inputs)[0]

        # Create masks for each section: tf.concat() used to break input into masked and non-masked segments then concat them back together
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

        # Select the mask for the current batch
        mask = masks[self.mask_index]

        # Apply the mask to the input data
        masked_inputs = tf.where(mask != 0, mask, inputs)

        return masked_inputs, mask


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

        # Compute binary cross-entropy loss without creating an instance
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
        # whole loss (no masking)
        whole_loss = tf.reduce_mean(loss)

        return tf.reduce_mean(masked_loss), whole_loss


class VAEWithCustomLoss(tf.keras.Model):
    def __init__(self, encoder, decoder, mask_index, **kwargs):
        super(VAEWithCustomLoss, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss_layer = CustomLossLayer()
        self.masking_layer = FixedMaskingLayer(mask_index=mask_index)

    def call(self, inputs, training=None):
        tf.print("\nStarting batch processing...")
        masked_inputs, mask = self.masking_layer(inputs)
        codings_mean, codings_log_var, codings = self.encoder(masked_inputs)
        reconstructions = self.decoder(codings)

        # KL Loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.square(codings_mean) - tf.exp(codings_log_var),
            axis=1,
        )
        kl_loss = tf.reduce_mean(kl_loss)

        # reconstruction loss (masked and whole)
        masked_reconstruction_loss, whole_reconstruction_loss = self.custom_loss_layer(
            inputs, reconstructions, mask
        )

        # Total losses (KL + reconstruction)
        total_masked_loss = masked_reconstruction_loss + kl_loss
        total_whole_loss = whole_reconstruction_loss + kl_loss

        if training:  # training on total loss
            self.add_loss(total_whole_loss)
        tf.print(" - Selected Subset (Fixed):", self.masking_layer.mask_index)
        tf.print("Finished batch processing...")
        return reconstructions


# Define the encoder
mask_section = 0  # set to 0: DNAm, 1: K9, 2: K27, 3: expression
codings_size = 50
inputs = tf.keras.layers.Input(shape=[12001])
masked_inputs, mask = FixedMaskingLayer(mask_index=mask_section)(inputs)
x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(
    masked_inputs
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
vae_model = VAEWithCustomLoss(
    encoder=variational_encoder, decoder=variational_decoder, mask_index=mask_section
)


### METRIC TRACKING
# Custom Metric Functions
def masked_total_loss_metric(y_true, y_pred):
    # Calculate the masked total loss (masked reconstruction + KL loss)
    masked_reconstruction_loss = custom_loss_layer(
        y_true, y_pred, mask=True
    )  # Placeholder for masked loss
    kl_loss = calculate_kl_loss(y_pred)  # Placeholder for KL loss
    return masked_reconstruction_loss + kl_loss


def whole_total_loss_metric(y_true, y_pred):
    # Calculate the total loss for the whole input (reconstruction + KL loss)
    whole_reconstruction_loss = custom_loss_layer(
        y_true, y_pred, mask=False
    )  # Placeholder for whole loss
    kl_loss = calculate_kl_loss(y_pred)  # Placeholder for KL loss
    return whole_reconstruction_loss + kl_loss


def masked_reconstruction_loss_metric(y_true, y_pred):
    # Use the CustomLossLayer to calculate the masked reconstruction loss
    custom_loss_layer = CustomLossLayer()
    masked_loss, _ = custom_loss_layer.call(
        y_true, y_pred, mask=True
    )  # Assuming mask=True is handled elsewhere
    return masked_loss


# Define a function for whole reconstruction loss
def whole_reconstruction_loss_metric(y_true, y_pred):
    custom_loss_layer = CustomLossLayer()
    _, whole_loss = custom_loss_layer.call(
        y_true, y_pred, mask=False
    )  # Assuming mask=False is handled elsewhere
    return whole_loss


def kl_loss_metric(y_true, y_pred):
    return calculate_kl_loss(y_pred)  # Placeholder for KL loss calculation


vae_model.compile(optimizer="adam")

# Train the model
history = vae_model.fit(
    X_train,
    X_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_val, X_val),
    verbose=1,
)

# Plot training and validation loss
# Create a figure with 2 subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
section_names = {0: "DNAm", 1: "K9", 2: "K27", 3: "expression"}
section_name = section_names[mask_section]

# Plot 1: Regular loss plot
axs[0].plot(history.history["loss"], label="Training Loss")
axs[0].plot(history.history["val_loss"], label="Validation Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()
axs[0].set_title("{} Training and Validation Loss".format(section_name))

# Plot 2: Loss plot with log y-axis
axs[1].plot(history.history["loss"], label="Training Loss")
axs[1].plot(history.history["val_loss"], label="Validation Loss")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss (Log Scale)")
axs[1].set_yscale("log")
axs[1].legend()
axs[1].set_title("{} Training and Validation Loss (Log Scale)".format(section_name))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig("vae_mask_singlecat_{}_v1.png".format(section_name))
