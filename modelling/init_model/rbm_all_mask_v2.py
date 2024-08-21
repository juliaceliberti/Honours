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
        section_size = 4000 * 2  # Each original feature becomes 2 binary features

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


# Function to train RBM and track metrics
def train(rbm, dataloader, val_loader, n_iter, lr=0.1, k=1):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    reconstruction_errors = []
    val_errors = []
    val_masked_errors = []

    for i in range(n_iter):
        batch_errors = []
        for batch in dataloader:
            data = batch[0]
            encoded_masked, masked_indices = encode_and_mask_batch(data)

            v0, v_prob, h_sample = rbm.contrastive_divergence(encoded_masked, k)

            error = rbm.reconstruction_error(encoded_masked, v_prob)
            batch_errors.append(error.item())

            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        avg_error = np.mean(batch_errors)
        reconstruction_errors.append(avg_error)

        # Evaluate on validation set
        val_error, val_masked_error = evaluate_rbm_with_error(rbm, val_loader)
        val_errors.append(val_error)
        val_masked_errors.append(val_masked_error)

        print(
            f"Iteration {i + 1}/{n_iter}, Train Error: {avg_error}, Validation Error: {val_error}, Validation Masked Error: {val_masked_error}"
        )

    return reconstruction_errors, val_errors, val_masked_errors


def evaluate_rbm_with_error(rbm, data_loader):
    whole_errors = []
    masked_errors = []
    for batch in data_loader:
        data = batch[0]
        encoded_masked, masked_indices = encode_and_mask_batch(data)

        v_prob = rbm(encoded_masked)
        error = rbm.reconstruction_error(encoded_masked, v_prob)
        masked_error = rbm.reconstruction_error_masked(
            encoded_masked, v_prob, masked_indices
        )
        whole_errors.append(error.item())
        masked_errors.append(masked_error.item())

    return np.mean(whole_errors), np.mean(masked_errors)


# Initialize and train the RBM
n_visible = X_train.shape[1] * 2  # since we are doubling the features
n_hidden = 256

rbm = RBM(n_visible, n_hidden)

n_iterations = 100
reconstruction_errors, val_errors, val_masked_errors = train(
    rbm, train_loader, val_loader, n_iterations
)

# Final evaluation on the test set
test_whole_error, test_masked_error = evaluate_rbm_with_error(rbm, test_loader)
print(
    f"Final Test Whole Error: {test_whole_error}, Final Test Masked Error: {test_masked_error}"
)

# Save the trained model
torch.save(rbm.state_dict(), "rbm_with_encoding_and_masking.pth")

# Save the reconstruction error plot
plt.plot(reconstruction_errors, label="Training Error")
plt.plot(val_errors, label="Validation Error")
plt.plot(val_masked_errors, label="Validation Masked Error")
plt.xlabel("Iterations")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error over 100 Iterations")
plt.legend()
plt.savefig("reconstruction_error_with_encoding_and_masking.png")
