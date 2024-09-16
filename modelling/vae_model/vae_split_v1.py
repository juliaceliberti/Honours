import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
)  # Convert expression to binary

# Concatenate all features except expression to make 1D (12000 features)
X = np.concatenate((dnam_features, h3k9me3_features, h3k27me3_features), axis=1)

# Separate data into silent and non-silent based on RNA expression
silent_indices = np.where(rna_expression == 0)[0]
non_silent_indices = np.where(rna_expression > 0)[0]

X_silent = X[silent_indices]
X_non_silent = X[non_silent_indices]

# Split each group into training, validation, and testing
X_train_silent, X_temp_silent = train_test_split(
    X_silent, test_size=0.3, random_state=42
)
X_val_silent, X_test_silent = train_test_split(
    X_temp_silent, test_size=0.66, random_state=42
)

X_train_non_silent, X_temp_non_silent = train_test_split(
    X_non_silent, test_size=0.3, random_state=42
)
X_val_non_silent, X_test_non_silent = train_test_split(
    X_temp_non_silent, test_size=0.66, random_state=42
)


### VAE Components
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
        self.add_loss(tf.reduce_mean(kl_loss) / 12000.0)
        return inputs


def create_vae_model(input_shape, codings_size):
    # ENCODER
    inputs = tf.keras.layers.Input(shape=[input_shape])
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
    x = tf.keras.layers.Dense(input_shape, activation="sigmoid")(
        x
    )  # Sigmoid for binary

    # output match original input
    variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[x])
    _, _, codings = variational_encoder(inputs)  # generate codings
    reconstructions = variational_decoder(codings)  # decode codings

    # Define the VAE model
    variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

    # compile
    variational_ae.compile(loss="binary_crossentropy", optimizer="adam")

    return variational_ae


# Define and train the VAE for silent data
vae_silent = create_vae_model(input_shape=12000, codings_size=50)
history_silent = vae_silent.fit(
    X_train_silent,
    X_train_silent,
    epochs=25,
    batch_size=128,
    validation_data=(X_val_silent, X_val_silent),
)

# Define and train the VAE for non-silent data
vae_non_silent = create_vae_model(input_shape=12000, codings_size=50)
history_non_silent = vae_non_silent.fit(
    X_train_non_silent,
    X_train_non_silent,
    epochs=25,
    batch_size=128,
    validation_data=(X_val_non_silent, X_val_non_silent),
)

# Plotting the training and validation loss for both VAEs
plt.figure(figsize=(14, 6))

# Plot for silent data VAE
plt.subplot(1, 2, 1)
plt.plot(history_silent.history["loss"], label="Training Loss")
plt.plot(history_silent.history["val_loss"], label="Validation Loss")
plt.title("VAE Training and Validation Loss (Silent Data)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot for non-silent data VAE
plt.subplot(1, 2, 2)
plt.plot(history_non_silent.history["loss"], label="Training Loss")
plt.plot(history_non_silent.history["val_loss"], label="Validation Loss")
plt.title("VAE Training and Validation Loss (Non-Silent Data)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("vae_silent_vs_non_silent.png")
