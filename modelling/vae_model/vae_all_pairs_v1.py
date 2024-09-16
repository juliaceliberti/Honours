import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# section to be masked
mask_section = 0  # set to 0: DNAm, 1: K9, 2: K27, 3: expression

# Set random seeds for reproducibility
seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

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
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)


### Model definition
class FixedMaskingLayer(
    tf.keras.layers.Layer
):  # masks a single subset of the input data (passed into the VAE class)
    def __init__(self, mask_index, dummy_index, **kwargs):
        super(FixedMaskingLayer, self).__init__(**kwargs)
        self.mask_index = mask_index  # Fixed subset to mask
        self.dummy_index = dummy_index

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

        # Select the mask and for the current batch
        mask = masks[self.mask_index]

        # select the mask for the dummy mask and convert from -1 to -2
        dummy_mask = masks[self.dummy_index]
        dummy_mask = tf.where(dummy_mask == -1.0, -2.0, dummy_mask)

        # Apply the mask to the input data
        masked_inputs = tf.where(mask != 0, mask, inputs)

        # apply dummy mask (we now have an input with one section masked with -1 and another masked with -2)
        dummy_masked_inputs = tf.where(dummy_mask != 0, dummy_mask, masked_inputs)

        # return the input with both masks and the primary mask (for masked reconstruction loss which doesn't include the dummy mask)
        return dummy_masked_inputs, mask


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, dummy_index, **kwargs):
        super().__init__(**kwargs)
        # model
        self.encoder = encoder
        self.decoder = decoder
        self.dummy_index = dummy_index
        # metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="val_recon_loss"
        )
        self.val_kl_loss_tracker = tf.keras.metrics.Mean(name="val_kl_loss")

    # access metrics like attributes
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,
        ]

    def call(self, inputs):
        dummy_masked_inputs, mask = FixedMaskingLayer(
            mask_index=mask_section, dummy_index=self.dummy_index
        )(inputs)
        codings_mean, codings_log_var, codings = self.encoder(dummy_masked_inputs)
        reconstruction = self.decoder(codings)
        return codings_mean, codings_log_var, reconstruction, mask

    # override train step to track KL and recon loss separately
    def train_step(self, data):

        x, _ = data

        # using a gradient tape to track the operation for differentiation
        with tf.GradientTape() as tape:
            # pass input through model to get latent codings, reconstruction and random mask
            codings_mean, codings_log_var, reconstruction, mask = self(x)

            # get reconstruction loss for masked section
            reconstruction_loss = self.compute_reconstruction_loss(
                x, reconstruction, mask
            )
            # calc KL loss
            kl_loss = self.compute_kl_loss(codings_mean, codings_log_var)
            # combine recon and KL loss for total loss
            total_loss = reconstruction_loss + kl_loss

        # compute gradients of the total loss then update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # update metrics for current batch
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # override test step to track KL and recon loss separately
    def test_step(self, data):
        x, _ = data
        codings_mean, codings_log_var, reconstruction, mask = self(x, training=False)
        reconstruction_loss = self.compute_reconstruction_loss(x, reconstruction, mask)
        kl_loss = self.compute_kl_loss(codings_mean, codings_log_var)
        total_loss = reconstruction_loss + kl_loss

        self.val_total_loss_tracker.update_state(total_loss)
        self.val_reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.val_total_loss_tracker.result(),
            "recon_loss": self.val_reconstruction_loss_tracker.result(),
            "kl_loss": self.val_kl_loss_tracker.result(),
        }

    # calc recontruction loss for the masked input section
    def compute_reconstruction_loss(self, inputs, reconstructions, mask):
        mask_indices = tf.cast(mask == -1, tf.float32)
        reconstructions_clipped = tf.clip_by_value(reconstructions, 1e-7, 1 - 1e-7)
        # use .backend not .losses as there aren't automatic reductions
        loss = tf.keras.backend.binary_crossentropy(inputs, reconstructions_clipped)
        masked_loss = tf.reduce_sum(loss * mask_indices, axis=1) / tf.reduce_sum(
            mask_indices, axis=1
        )
        # avoid division by zero
        masked_loss = tf.where(
            tf.reduce_sum(mask_indices, axis=1) == 0, 0.0, masked_loss
        )

        return tf.reduce_mean(masked_loss)

    # calc KL Loss
    def compute_kl_loss(self, codings_mean, codings_log_var):
        kl_loss = -0.5 * tf.reduce_sum(
            1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
            axis=-1,
        )
        return tf.reduce_mean(kl_loss)

    # reset metrics to avoid accumulation over batches
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()


# Function to train VAE with a primary mask and a dummy mask (trains one model)
def train_with_mask_pair(mask_section, dummy_section, X_train, X_val, section_names):
    # Define the encoder
    codings_size = 100
    inputs = tf.keras.layers.Input(shape=[12001])
    dummy_masked_inputs, mask = FixedMaskingLayer(
        mask_index=mask_section, dummy_index=dummy_section
    )(inputs)

    # Encoder layers
    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(
        dummy_masked_inputs
    )
    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    codings_mean = tf.keras.layers.Dense(codings_size)(x)  # mean
    codings_log_var = tf.keras.layers.Dense(codings_size)(x)  # log_var
    codings = Sampling()([codings_mean, codings_log_var])
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

    # Compile model
    vae_model = VAE(
        encoder=variational_encoder,
        decoder=variational_decoder,
        dummy_index=dummy_section,
    )
    vae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train model
    history = vae_model.fit(
        X_train,
        X_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=1,
    )

    # Return the history and the section names involved
    return history, section_names[mask_section], section_names[dummy_section]


# Perform the all-pairs study dynamically (uses the training functiont to train all 3 models)
def perform_all_pairs_study(mask_section, X_train, X_val):
    section_names = {0: "DNAm", 1: "K9", 2: "K27", 3: "expression"}

    # Get the dummy masks (those that aren't the primary mask)
    dummy_masks = [i for i in range(4) if i != mask_section]

    all_histories = []

    # Loop through dummy masks and train with each pair
    for dummy_section in dummy_masks:
        print(
            f"Training with primary mask: {section_names[mask_section]}, dummy mask: {section_names[dummy_section]}"
        )

        history, primary_section, dummy_section = train_with_mask_pair(
            mask_section, dummy_section, X_train, X_val, section_names
        )

        # Store history and section names for analysis later
        all_histories.append((history, primary_section, dummy_section))

    return all_histories


all_histories = perform_all_pairs_study(mask_section, X_train, X_val)


# Function to plot the training and validation loss for all-pairs study
def plot_all_pairs_losses(all_histories, mask_section, section_names):
    # Create subplots for regular and log scale
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))  # Regular scale plot
    fig_log, axs_log = plt.subplots(1, 3, figsize=(18, 5))  # Log scale plot

    # Define fixed colors for different dummy masks (red, blue, green)
    colors = ["red", "blue", "green"]

    # Loop through all histories and plot
    for i, (history, primary_name, dummy_name) in enumerate(all_histories):
        # Extract the history data for current run
        history_dict = history.history
        epochs = range(
            1, len(history_dict["loss"]) + 1
        )  # Ensure line stops after the model's training epochs

        # Get the color for this dummy mask
        color = colors[i % len(colors)]

        # Total loss plot
        axs[0].plot(
            epochs,
            history_dict["loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs[0].plot(
            epochs,
            history_dict["val_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )

        # Reconstruction loss plot
        axs[1].plot(
            epochs,
            history_dict["recon_loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs[1].plot(
            epochs,
            history_dict["val_recon_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )

        # KL loss plot
        axs[2].plot(
            epochs,
            history_dict["kl_loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs[2].plot(
            epochs,
            history_dict["val_kl_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )

        # Repeat the same for log scale plot
        axs_log[0].plot(
            epochs,
            history_dict["loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs_log[0].plot(
            epochs,
            history_dict["val_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )
        axs_log[1].plot(
            epochs,
            history_dict["recon_loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs_log[1].plot(
            epochs,
            history_dict["val_recon_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )
        axs_log[2].plot(
            epochs,
            history_dict["kl_loss"],
            label=f"Train (Dummy = {dummy_name})",
            color=color,
        )
        axs_log[2].plot(
            epochs,
            history_dict["val_kl_loss"],
            label=f"Validation (Dummy = {dummy_name})",
            linestyle="--",
            color=color,
        )

    # Set titles and labels for regular scale plot
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Total Loss")
    axs[0].legend()
    axs[0].set_title("Total Loss")

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Reconstruction Loss")
    axs[1].legend()
    axs[1].set_title("Reconstruction Loss")

    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("KL Loss")
    axs[2].legend()
    axs[2].set_title("KL Loss")

    # Set titles and labels for log scale plot
    axs_log[0].set_xlabel("Epochs")
    axs_log[0].set_ylabel("Total Loss (Log Scale)")
    axs_log[0].set_yscale("log")
    axs_log[0].legend()
    axs_log[0].set_title("Total Loss (Log Scale)")

    axs_log[1].set_xlabel("Epochs")
    axs_log[1].set_ylabel("Reconstruction Loss (Log Scale)")
    axs_log[1].set_yscale("log")
    axs_log[1].legend()
    axs_log[1].set_title("Reconstruction Loss (Log Scale)")

    axs_log[2].set_xlabel("Epochs")
    axs_log[2].set_ylabel("KL Loss (Log Scale)")
    axs_log[2].set_yscale("log")
    axs_log[2].legend()
    axs_log[2].set_title("KL Loss (Log Scale)")

    # Set the overall titles for the entire figure
    section_name = section_names[mask_section]
    fig.suptitle(f"VAE Training and Validation Losses (Primary Mask = {section_name})")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the regular scale plot
    fig.savefig(f"vae_loss_plot_{section_name}_all_pairs_v4.png")

    # Log scale plot
    fig_log.suptitle(
        f"VAE Training and Validation Losses (Primary Mask = {section_name}) - Log Scale"
    )
    fig_log.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the log scale plot
    fig_log.savefig(f"vae_loss_plot_{section_name}_all_pairs_v4_log.png")


# Call the plotting function after training all models
section_names = {0: "DNAm", 1: "K9", 2: "K27", 3: "expression"}
plot_all_pairs_losses(all_histories, mask_section, section_names)
