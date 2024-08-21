import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import mean_squared_error

### DATA PREPARATION

# Load data
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
expression_values = rna_expression_df["expression"].values.reshape(
    -1, 1
)  # to make 2D like other features

# Concatenate all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)


# Split into training, validation, and testing (70% training, 10% validation, 20% testing)
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.66, random_state=42)

### Define DataLoader to handle batching
batch_size = 32
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

### Function to encode and mask data in batches


def encode_and_mask_batch(batch):
    def encode_data(X_batch):
        # Create an empty array to store encoded data
        n_samples, n_features = X_batch.shape
        encoded_X = np.zeros(
            (n_samples, n_features * 2)
        )  # Each original feature becomes 2 binary features

        # Define encoding
        unmodified = np.array([0, 1])
        modified = np.array([1, 0])

        # Populate the encoded_X array
        for i in range(n_samples):
            for j in range(n_features):
                if X_batch[i, j] == 0:  # Unmodified
                    encoded_X[i, j * 2 : j * 2 + 2] = unmodified
                elif X_batch[i, j] == 1:  # Modified
                    encoded_X[i, j * 2 : j * 2 + 2] = modified

        return encoded_X

    def apply_mask(data):
        """
        Randomly select a section of the data to mask.
        Args:
            data (numpy array): Encoded data.
        Returns:
            masked_data (numpy array): Data with a section masked.
        """
        masked_data = data.copy()
        n_samples, n_features = masked_data.shape
        section_size = 4000 * 2  # Each original feature becomes 2 binary features

        # Randomly choose which section to mask: 0 for DNAm, 1 for H3K9me3, 2 for H3K27me3, 3 for expression
        section_to_mask = np.random.choice(4)

        if (
            section_to_mask < 3
        ):  # Mask one of the modification sections (DNAm, H3K9me3, H3K27me3)
            start_idx = section_to_mask * section_size
            end_idx = start_idx + section_size

            # Replicate the mask [1, 0] across the entire section
            mask = np.tile(np.array([1, 0]), section_size // 2)
            masked_data[:, start_idx:end_idx] = mask
        else:  # Mask the expression value
            start_idx = -2
            end_idx = None
            masked_data[:, start_idx:end_idx] = np.array(
                [0, 0]
            )  # Mask the last feature (expression)

        return masked_data, (start_idx, end_idx)

    # Encode the batch
    encoded_batch = encode_data(batch)

    # Apply masking
    masked_batch = apply_mask(encoded_batch)

    return masked_batch


def evaluate_reconstruction_error(original_data, reconstructed_data, masked_indices):
    """
    Evaluate reconstruction error on the whole input and the masked section.

    Args:
        original_data (numpy array): The original input data before masking.
        reconstructed_data (numpy array): The reconstructed data from the RBM.
        masked_indices (tuple): A tuple (start_idx, end_idx) indicating the indices of the masked section.

    Returns:
        whole_error (float): The reconstruction error on the whole input.
        masked_error (float): The reconstruction error on the masked section only.
    """
    # Calculate error on the whole input
    whole_error = mean_squared_error(original_data, reconstructed_data)

    # Calculate error on the masked section only
    start_idx, end_idx = masked_indices
    masked_error = mean_squared_error(
        original_data[:, start_idx:end_idx], reconstructed_data[:, start_idx:end_idx]
    )

    return whole_error, masked_error


def evaluate_rbm_with_error(rbm, data_loader):
    whole_errors = []
    masked_errors = []
    for batch in data_loader:
        batch_data = batch[0].numpy()
        batch_encoded_masked, masked_indices = encode_and_mask_batch(batch_data)

        # Get hidden layer activations
        hidden_activations = rbm.transform(batch_encoded_masked)
        # Reconstruct the input from the hidden activations
        reconstructed_batch = rbm.inverse_transform(hidden_activations)

        # Evaluate reconstruction error
        whole_error, masked_error = evaluate_reconstruction_error(
            batch_data, reconstructed_batch, masked_indices
        )
        whole_errors.append(whole_error)
        masked_errors.append(masked_error)

    # Average errors over all batches
    avg_whole_error = np.mean(whole_errors)
    avg_masked_error = np.mean(masked_errors)

    return avg_whole_error, avg_masked_error


### RBM TRAINING WITH BATCHING

# Initialize the RBM model
rbm = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=10, random_state=0)

# Train the RBM in batches
for batch_num, batch in enumerate(train_loader):
    batch_data = batch[0].numpy()  # Get the batch data as a NumPy array
    batch_encoded_masked, masked_indices = encode_and_mask_batch(batch_data)
    rbm.fit(batch_encoded_masked)

    # Evaluate on the validation set after each batch
    val_whole_error, val_masked_error = evaluate_rbm_with_error(rbm, val_loader)
    print(
        f"Batch {batch_num + 1}: Validation Whole Error: {val_whole_error}, Validation Masked Error: {val_masked_error}"
    )

### VALIDATION AND TESTING WITH BATCHING AND ERROR EVALUATION

# Final evaluation on test set
test_whole_error, test_masked_error = evaluate_rbm_with_error(rbm, test_loader)
print(
    f"Final Test Whole Error: {test_whole_error}, Final Test Masked Error: {test_masked_error}"
)
