import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

### DATA PREPARATION
gene_matrix_array = np.load("../../../modelling/init_model/gene_matrix_list.npy")[:100]
rna_expression_df = pd.read_csv(
    "../../../modelling/init_model/rna_expression_list.csv"
).iloc[:100]
print("data loaded")

# Check order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

rna_expression = (rna_expression_df["expression"].values > 0).astype(int)


# Separate modification types
dnam_features = gene_matrix_array[:, :, 0]
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]
expression_values = rna_expression.reshape(-1, 1)

# Concatenate all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)

# Split into training, validation, and testing (70% training, 10% validation, 20% testing)
X_train, X_temp = train_test_split(X, test_size=0.3, random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.66, random_state=42)

# Convert data to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))

# Define DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


### RBM CLASS DEFINITION
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v):
        h_prob, h_sample = self.sample_h(v)
        v_prob, v_sample = self.sample_v(h_sample)
        return v_prob

    def contrastive_divergence(self, v, k=1):
        v0 = v
        for _ in range(k):
            h_prob, h_sample = self.sample_h(v0)
            v_prob, v0 = self.sample_v(h_sample)
        return v0, v_prob, h_sample

    def reconstruction_error(self, v, v_reconstructed):
        return torch.mean(torch.sum((v - v_reconstructed) ** 2, dim=1))

    def reconstruction_error_masked(self, v, v_reconstructed, masked_indices):
        start_idx, end_idx = masked_indices
        masked_v = v[:, start_idx:end_idx]
        masked_v_reconstructed = v_reconstructed[:, start_idx:end_idx]
        return torch.mean(torch.sum((masked_v - masked_v_reconstructed) ** 2, dim=1))


# Function to encode and mask data in batches
def encode_and_mask_batch(batch):
    def encode_data(X_batch):
        n_samples, n_features = X_batch.shape
        encoded_X = torch.zeros(
            (n_samples, n_features * 2)
        )  # Each original feature becomes 2 binary features

        unmodified = torch.tensor([0, 1], dtype=torch.float32)
        modified = torch.tensor([1, 0], dtype=torch.float32)

        for i in range(n_samples):
            for j in range(n_features):
                if X_batch[i, j] == 0:  # Unmodified
                    encoded_X[i, j * 2 : j * 2 + 2] = unmodified
                elif X_batch[i, j] == 1:  # Modified
                    encoded_X[i, j * 2 : j * 2 + 2] = modified

        return encoded_X

    def apply_mask(data):
        masked_data = data.clone()
        n_samples, n_features = masked_data.shape
        section_size = (n_features - 2) // 3  # Adjust section size for your data

        section_to_mask = np.random.choice(4)
        if section_to_mask < 3:
            start_idx = section_to_mask * section_size
            end_idx = start_idx + section_size
            mask = torch.tensor([1, 0], dtype=torch.float32).repeat(section_size // 2)
            masked_data[:, start_idx:end_idx] = mask
        else:
            start_idx = -2
            end_idx = None
            masked_data[:, start_idx:] = torch.tensor([0, 0], dtype=torch.float32)

        return masked_data, (start_idx, end_idx)

    encoded_batch = encode_data(batch)
    masked_batch, masked_indices = apply_mask(encoded_batch)

    return masked_batch, masked_indices


print("loading model")
# Load the trained model
rbm = RBM(n_visible=X_train.shape[1] * 2, n_hidden=256)  # Match model architecture
rbm.load_state_dict(torch.load("rbm_with_encoding_and_masking.pth"))
rbm.eval()  # Set the model to evaluation mode

# Select a single input from the test set
single_input = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(
    0
)  # Add batch dimension

# Encode and mask the input
encoded_input = encode_and_mask_batch(single_input.numpy())[0]

# Pass the masked input through the RBM
with torch.no_grad():  # Disable gradient calculation for inference
    reconstructed_input = rbm(encoded_input)


# Decode the data to compare
def decode_data(encoded_X):
    n_samples, n_features = encoded_X.shape
    decoded_X = torch.zeros(
        (n_samples, n_features // 2)
    )  # Each pair of binary features becomes one original feature

    for i in range(n_samples):
        for j in range(0, n_features, 2):
            if torch.equal(
                encoded_X[i, j : j + 2], torch.tensor([1, 0], dtype=torch.float32)
            ):
                decoded_X[i, j // 2] = 1  # Modified
            elif torch.equal(
                encoded_X[i, j : j + 2], torch.tensor([0, 1], dtype=torch.float32)
            ):
                decoded_X[i, j // 2] = 0  # Unmodified

    return decoded_X


# Decode the original, masked, and reconstructed inputs
decoded_original_input = single_input
decoded_masked_input = decode_data(encoded_input)
decoded_reconstructed_input = decode_data(reconstructed_input)

# Convert to NumPy arrays for easy comparison
original_input_np = decoded_original_input.squeeze(0).numpy()
masked_input_np = decoded_masked_input.squeeze(0).numpy()
reconstructed_input_np = decoded_reconstructed_input.squeeze(0).numpy()

import matplotlib.pyplot as plt

# Assuming the first 12000 features represent the modification types, and the last feature is expression

# Visualize the first 12000 features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(
    original_input_np[:12000].reshape(100, 120)
)  # Reshape to something that fits
plt.title("Original Input")

plt.subplot(1, 3, 2)
plt.imshow(masked_input_np[:12000].reshape(100, 120))  # Reshape accordingly
plt.title("Masked Input")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_input_np[:12000].reshape(100, 120))  # Reshape accordingly
plt.title("Reconstructed Input")

plt.show()

# Print the last feature (expression value) separately
print("\nOriginal Expression Value:", original_input_np[-1])
print("Masked Expression Value:", masked_input_np[-1])
print("Reconstructed Expression Value:", reconstructed_input_np[-1])
