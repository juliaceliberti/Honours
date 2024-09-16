import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### DATA PREPARATION (As before)
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
)  # Convert expression to binary
expression_values = rna_expression.reshape(-1, 1)  # Shape (58780, 1)

# Concatenate all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)

# Split into training, validation, and testing (70% training, 10% validation, 20% testing)
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.66, random_state=42)


### MASKING LAYER
### MASKING LAYER
class MaskingLayer(tf.keras.layers.Layer):
    def __init__(self, mask_value=-1):
        super(MaskingLayer, self).__init__()
        self.mask_value = mask_value

    def call(self, inputs, training=None):
        mask_choice = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)

        if training:
            mask_choice = tf.cast(mask_choice, dtype=tf.int32)

            inputs = tf.cond(
                mask_choice == 0,
                lambda: tf.concat(
                    [
                        tf.cast(
                            tf.fill([tf.shape(inputs)[0], 4000], self.mask_value),
                            dtype=tf.float32,
                        ),
                        inputs[:, 4000:],
                    ],
                    axis=1,
                ),
                lambda: inputs,
            )
            inputs = tf.cond(
                mask_choice == 1,
                lambda: tf.concat(
                    [
                        inputs[:, :4000],
                        tf.cast(
                            tf.fill([tf.shape(inputs)[0], 4000], self.mask_value),
                            dtype=tf.float32,
                        ),
                        inputs[:, 8000:],
                    ],
                    axis=1,
                ),
                lambda: inputs,
            )
            inputs = tf.cond(
                mask_choice == 2,
                lambda: tf.concat(
                    [
                        inputs[:, :8000],
                        tf.cast(
                            tf.fill([tf.shape(inputs)[0], 4000], self.mask_value),
                            dtype=tf.float32,
                        ),
                        inputs[:, 12000:],
                    ],
                    axis=1,
                ),
                lambda: inputs,
            )
            inputs = tf.cond(
                mask_choice == 3,
                lambda: tf.concat(
                    [
                        inputs[:, :12000],
                        tf.cast(
                            tf.fill([tf.shape(inputs)[0], 1], self.mask_value),
                            dtype=tf.float32,
                        ),
                    ],
                    axis=1,
                ),
                lambda: inputs,
            )

        return inputs, mask_choice


### Custom Loss Layer
class CustomLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)

    def call(self, y_true, y_pred, mask_choice):
        # Mask sections based on mask_choice
        if mask_choice == 0:
            y_true_masked = y_true[:, :4000]
            y_pred_masked = y_pred[:, :4000]
        elif mask_choice == 1:
            y_true_masked = y_true[:, 4000:8000]
            y_pred_masked = y_pred[:, 4000:8000]
        elif mask_choice == 2:
            y_true_masked = y_true[:, 8000:12000]
            y_pred_masked = y_pred[:, 8000:12000]
        else:
            y_true_masked = y_true[:, 12000:12001]
            y_pred_masked = y_pred[:, 12000:12001]

        # Clip predictions to prevent log(0) or log(negative number) issues
        y_pred_masked = tf.clip_by_value(
            y_pred_masked, clip_value_min=1e-7, clip_value_max=1 - 1e-7
        )
        loss = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred_masked)

        return tf.reduce_mean(loss)  # Return the mean loss across the batch


### VAE Components (Sampling and KL Divergence Layers)
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


def create_vae_model(input_shape, codings_size, mask_value=-1):
    # ENCODER
    inputs = tf.keras.layers.Input(shape=[input_shape])
    masked_inputs, mask_choice = MaskingLayer(mask_value=mask_value)(inputs)
    x = tf.keras.layers.Dense(512, activation="relu")(masked_inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    codings_mean = tf.keras.layers.Dense(codings_size)(x)  # mean
    codings_log_var = tf.keras.layers.Dense(codings_size)(x)  # log_var
    codings = Sampling()([codings_mean, codings_log_var])

    # KL Divergence Layer
    codings_mean, codings_log_var = KLDivergenceLayer()([codings_mean, codings_log_var])

    # Define VAE Encoder
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

    # Define VAE Decoder
    variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[x])
    _, _, codings = variational_encoder(inputs)  # Generate codings
    reconstructions = variational_decoder(codings)  # Decode codings

    # Calculate loss using custom loss layer
    loss_layer = CustomLossLayer()
    loss_value = loss_layer(inputs, reconstructions, mask_choice)

    # Define the VAE model
    variational_ae = tf.keras.Model(inputs=[inputs], outputs=[reconstructions])

    # Add the KL Divergence loss and compile the model
    variational_ae.add_loss(loss_value)  # Custom reconstruction loss
    variational_ae.compile(optimizer="adam")

    return variational_ae


# Define and train the VAE
vae = create_vae_model(input_shape=12001, codings_size=50, mask_value=-1)
history = vae.fit(
    X_train, X_train, epochs=150, batch_size=128, validation_data=(X_val, X_val)
)

# Plotting the training and validation loss
# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# First subplot: Regular Loss Plot
axs[0].plot(history.history["loss"], label="Training Loss")
axs[0].plot(history.history["val_loss"], label="Validation Loss")
axs[0].set_title("VAE Training and Validation Loss")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("Loss")
axs[0].legend()

# Second subplot: Log-scaled Loss Plot
axs[1].plot(history.history["loss"], label="Training Loss")
axs[1].plot(history.history["val_loss"], label="Validation Loss")
axs[1].set_yscale("log")  # Set y-axis to log scale
axs[1].set_title("VAE Training and Validation Loss (Log Scale)")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("Loss (Log Scale)")
axs[1].legend()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("vae_with_masking_and_reconstruction_subplots.png")
plt.show()
