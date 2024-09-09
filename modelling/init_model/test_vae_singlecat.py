import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

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


### Model definition
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
    def call(self, inputs, reconstructions, mask):
        mask_indices = tf.cast(mask == -1, tf.float32)
        masked_loss = self.compute_loss(inputs, reconstructions, mask_indices)
        whole_loss = self.compute_loss(
            inputs, reconstructions, tf.ones_like(mask_indices)
        )
        return masked_loss, whole_loss

    def compute_loss(self, inputs, reconstructions, mask_indices):
        reconstructions_clipped = tf.clip_by_value(reconstructions, 1e-7, 1 - 1e-7)
        loss = tf.keras.backend.binary_crossentropy(inputs, reconstructions_clipped)
        loss = tf.reshape(loss, tf.shape(inputs))
        masked_loss = tf.reduce_sum(loss * mask_indices, axis=1) / tf.reduce_sum(
            mask_indices, axis=1
        )
        masked_loss = tf.where(
            tf.reduce_sum(mask_indices, axis=1) == 0, 0.0, masked_loss
        )  # avoid division by zero error
        return tf.reduce_mean(masked_loss)


class VAEWithCustomLoss(tf.keras.Model):
    def __init__(self, encoder, decoder, mask_index, **kwargs):
        super(VAEWithCustomLoss, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.custom_loss_layer = CustomLossLayer()
        self.masking_layer = FixedMaskingLayer(mask_index=mask_index)

        # Custom metrics for training and validation - averaged for each training step
        self.train_masked_loss_metric = tf.keras.metrics.Mean(name="train_masked_loss")
        self.train_whole_loss_metric = tf.keras.metrics.Mean(name="train_whole_loss")
        self.train_kl_loss_metric = tf.keras.metrics.Mean(name="train_kl_loss")
        self.masked_loss_metric = tf.keras.metrics.Mean(name="masked_loss")
        self.whole_loss_metric = tf.keras.metrics.Mean(name="whole_loss")
        self.kl_loss_metric = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None):
        masked_inputs, mask = self.masking_layer(inputs)
        codings_mean, codings_log_var, codings = self.encoder(masked_inputs)
        reconstructions = self.decoder(codings)

        masked_loss, whole_loss = self.custom_loss_layer(inputs, reconstructions, mask)
        self.add_loss(masked_loss)
        return reconstructions, masked_loss, whole_loss

        # customise the fit method

    # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            reconstructions, masked_loss, whole_loss = self(
                x, training=True
            )  # Get loss from call()
            loss = tf.reduce_sum(
                self.losses
            )  # use the KL Loss and Masked loss for total loss

            kl_loss = loss - masked_loss

        # Apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update custom metrics with the already-calculated masked and whole loss
        self.train_masked_loss_metric.update_state(masked_loss)
        self.train_whole_loss_metric.update_state(whole_loss)
        self.train_kl_loss_metric.update_state(kl_loss)

        return {
            "loss": loss,
            "train_masked_loss": self.train_masked_loss_metric.result(),
            "train_whole_loss": self.train_whole_loss_metric.result(),
            "train_kl_loss": self.train_kl_loss_metric.result(),
        }

    def test_step(self, data):
        x, _ = data
        reconstructions, masked_loss, whole_loss = self(
            x, training=False
        )  # Get loss from call()
        loss = tf.reduce_sum(
            self.losses
        )  # use the KL Loss and Masked loss for total loss
        kl_loss = loss - masked_loss

        # Update custom metrics with the already-calculated masked and whole loss
        self.masked_loss_metric.update_state(masked_loss)
        self.whole_loss_metric.update_state(whole_loss)
        self.kl_loss_metric.update_state(kl_loss)

        return {
            "loss": loss,
            "masked_loss": self.masked_loss_metric.result(),
            "whole_loss": self.whole_loss_metric.result(),
            "kl_loss": self.kl_loss_metric.result(),
        }


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


# Compile the VAE model
vae_model = VAEWithCustomLoss(
    encoder=variational_encoder, decoder=variational_decoder, mask_index=mask_section
)
vae_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
)

# Adding early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)


# Train the model
history = vae_model.fit(
    X_train,
    X_train,
    epochs=15,
    batch_size=128,
    validation_data=(X_val, X_val),
    callbacks=[early_stopping],
    verbose=1,
)


# Plot training and validation loss
# Create a figure with 2 subplots (1 row, 2 columns)
section_names = {0: "DNAm", 1: "K9", 2: "K27", 3: "expression"}
section_name = section_names[mask_section]

# get keys out of history.history
history_dict = history.history

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(
    1, len(history_dict["train_whole_loss"]) + 1
)  # start at 1 as epochs in keras start at 1

# Plot 1: breakdown of reconstrcution loss into training and validation, masking and whole
# Regular scale
axs[0].plot(
    epochs,
    history_dict["train_whole_loss"],
    label="Train Whole Loss",
    color="blue",
    linestyle="-",
)
axs[0].plot(
    epochs,
    history_dict["val_whole_loss"],
    label="Validation Whole Loss",
    color="orange",
    linestyle="-",
)
axs[0].plot(
    epochs,
    history_dict["train_masked_loss"],
    label="Train Masked Loss",
    color="blue",
    linestyle="--",
)
axs[0].plot(
    epochs,
    history_dict["val_masked_loss"],
    label="Validation Masked Loss",
    color="orange",
    linestyle="--",
)
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].set_title("Whole and Masked Losses ({}) (Regular Scale)".format(section_name))
axs[0].legend()

# Log scale
axs[1].plot(
    epochs,
    history_dict["train_whole_loss"],
    label="Train Whole Loss",
    color="blue",
    linestyle="-",
)
axs[1].plot(
    epochs,
    history_dict["val_whole_loss"],
    label="Validation Whole Loss",
    color="orange",
    linestyle="-",
)
axs[1].plot(
    epochs,
    history_dict["train_masked_loss"],
    label="Train Masked Loss",
    color="blue",
    linestyle="--",
)
axs[1].plot(
    epochs,
    history_dict["val_masked_loss"],
    label="Validation Masked Loss",
    color="orange",
    linestyle="--",
)
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss (Log Scale)")
axs[1].set_yscale("log")
axs[1].set_title("Whole and Masked Losses ({}) (Log Scale)".format(section_name))
axs[1].legend()

plt.tight_layout()
plt.savefig("vae_allcat_mask_vs_whole_loss_{}_v3.png".format(section_name))

# plot 2: breaking total loss down into training and validation, masked reconstruction and KL loss
# Plot KL Loss
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    epochs,
    history_dict["train_kl_loss"],
    label="Train KL Loss",
    color="green",
    linestyle="-",
)
ax.plot(
    epochs,
    history_dict["val_kl_loss"],
    label="Validation KL Loss",
    color="red",
    linestyle="-",
)
ax.plot(
    epochs,
    history_dict["train_masked_loss"],
    label="Train Masked Loss",
    color="green",
    linestyle="--",
)
ax.plot(
    epochs,
    history_dict["val_masked_loss"],
    label="Validation Masked Loss",
    color="red",
    linestyle="--",
)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("KL Loss Over Epochs ({})".format(section_name))
ax.legend()

plt.tight_layout()
plt.savefig("kl_loss_plot_{}_v3.png".format(section_name))
