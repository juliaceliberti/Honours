""""
This script uses a Restricted Boltzmann Machine on split silent and non-silent datasets 
and uses masking for both training and testing 
(to conclude how well the model can predict unseen data)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

### DATA PREPARATION
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# check order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# sep modification types
dnam_features = gene_matrix_array[:, :, 0]
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]

# concat features to make 1D
X = np.concatenate((dnam_features, h3k9me3_features, h3k27me3_features), axis=1)

# get expression values
expression_values = rna_expression_df["expression"].values

# separate data into silent and non-silent based on expression values
silent_indices = np.where(expression_values == 0)[0]  # Indices of silent genes
non_silent_indices = np.where(expression_values > 0)[0]  # Indices of non-silent genes

X_silent = X[silent_indices]  # Silent gene data
X_non_silent = X[non_silent_indices]  # Non-silent gene data

# convert data to PyTorch tensors
X_silent = torch.tensor(X_silent, dtype=torch.float32)  # Silent gene data
X_non_silent = torch.tensor(X_non_silent, dtype=torch.float32)  # Non-silent gene data

# split into training, validation, and test sets
X_silent_train, X_silent_temp = train_test_split(
    X_silent, test_size=0.4, random_state=42
)
X_silent_val, X_silent_test = train_test_split(
    X_silent_temp, test_size=0.5, random_state=42
)

X_non_silent_train, X_non_silent_temp = train_test_split(
    X_non_silent, test_size=0.4, random_state=42
)
X_non_silent_val, X_non_silent_test = train_test_split(
    X_non_silent_temp, test_size=0.5, random_state=42
)

# convert to TensorDatasets
train_dataset_silent = TensorDataset(X_silent_train)
val_dataset_silent = TensorDataset(X_silent_val)
test_dataset_silent = TensorDataset(X_silent_test)

train_dataset_non_silent = TensorDataset(X_non_silent_train)
val_dataset_non_silent = TensorDataset(X_non_silent_val)
test_dataset_non_silent = TensorDataset(X_non_silent_test)

# dataLoaders for batching
batch_size = 32
train_loader_silent = DataLoader(
    train_dataset_silent, batch_size=batch_size, shuffle=True
)
val_loader_silent = DataLoader(val_dataset_silent, batch_size=batch_size, shuffle=False)
test_loader_silent = DataLoader(
    test_dataset_silent, batch_size=batch_size, shuffle=False
)

train_loader_non_silent = DataLoader(
    train_dataset_non_silent, batch_size=batch_size, shuffle=True
)
val_loader_non_silent = DataLoader(
    val_dataset_non_silent, batch_size=batch_size, shuffle=False
)
test_loader_non_silent = DataLoader(
    test_dataset_non_silent, batch_size=batch_size, shuffle=False
)


### RBM CLASS DEFINITION
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        # two main layers
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weights and biases for hidden and visible layers
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        # get hidden layer class proba
        h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        # get visible layer class proba
        v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return v_prob, torch.bernoulli(v_prob)

    def forward(self, v, mask):
        # forward pass consists of calculating hidden proba then passing back to visible for v proba
        v = v * mask  # Ensure the model sees the masked data correctly
        h_prob, h_sample = self.sample_h(v)
        v_prob, v_sample = self.sample_v(h_sample)
        return v_prob

    def contrastive_divergence(self, v, k=1):
        # training method
        v0 = v
        for _ in range(k):
            h_prob, h_sample = self.sample_h(v0)
            v_prob, v0 = self.sample_v(h_sample)
        return v0, v_prob, h_sample

    def reconstruction_error(self, v, v_reconstructed):
        # calculate error
        return torch.mean(torch.sum((v - v_reconstructed) ** 2, dim=1))


# init RBM models
n_visible = X_silent.shape[1]
n_hidden = 256

rbm_silent = RBM(n_visible, n_hidden)
rbm_non_silent = RBM(n_visible, n_hidden)


# masking Function
def apply_mask(data, mask_ratio=0.2):
    mask = (torch.rand(data.size()) > mask_ratio).float()  # Create a binary mask
    masked_data = data * mask  # Apply the mask to the data
    return masked_data, mask


# train RBM and track metrics with batching and validation
def train(rbm, train_loader, val_loader, n_iter, lr=0.1, k=1, mask_ratio=0.2):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    train_errors = []
    val_errors = []

    for i in range(n_iter):
        # train
        batch_train_errors = []
        for batch in train_loader:
            data_train = batch[0]  # get batch
            masked_data_train, mask = apply_mask(
                data_train, mask_ratio
            )  # apply mask to training batch
            v0, v_prob, h_sample = rbm.contrastive_divergence(masked_data_train, k)

            # calc reconstruction error on batch
            train_error = rbm.reconstruction_error(data_train, v_prob)
            batch_train_errors.append(train_error.item())

            # update model weights
            optimizer.zero_grad()
            train_error.backward()
            optimizer.step()

        # get average of batch errors and add to train error list
        avg_train_error = np.mean(batch_train_errors)
        train_errors.append(avg_train_error)

        # validation
        batch_val_errors = []
        with torch.no_grad():  # no model updates for val
            for batch in val_loader:
                data_val = batch[0]
                masked_data_val, mask = apply_mask(
                    data_val, mask_ratio
                )  # mask val data
                v_prob_val = rbm(masked_data_val)
                val_error = rbm.reconstruction_error(data_val, v_prob_val)
                batch_val_errors.append(val_error.item())

        avg_val_error = np.mean(batch_val_errors)
        val_errors.append(avg_val_error)

        print(
            f"Iteration {i + 1}/{n_iter}, Train Error: {avg_train_error}, Validation Error: {avg_val_error}"
        )

    return train_errors, val_errors


# Train RBMs and track errors with batching
n_iterations = 100
train_errors_silent, val_errors_silent = train(
    rbm_silent,
    train_loader_silent,
    val_loader_silent,
    n_iterations,
    mask_ratio=mask_ratio,
)
train_errors_non_silent, val_errors_non_silent = train(
    rbm_non_silent,
    train_loader_non_silent,
    val_loader_non_silent,
    n_iterations,
    mask_ratio=mask_ratio,
)


# Testing the model on the test set
def test_rbm(rbm, test_loader, mask_ratio=0.2):
    test_errors = []
    with torch.no_grad():  # Disable gradient calculation for testing
        for batch in test_loader:
            data_test = batch[0]
            masked_data_test, mask = apply_mask(
                data_test, mask_ratio
            )  # Apply mask to test data
            v_prob_test = rbm(masked_data_test)
            test_error = rbm.reconstruction_error(data_test, v_prob_test)
            test_errors.append(test_error.item())

    avg_test_error = np.mean(test_errors)
    return avg_test_error


test_error_silent = test_rbm(rbm_silent, test_loader_silent, mask_ratio=mask_ratio)
test_error_non_silent = test_rbm(
    rbm_non_silent, test_loader_non_silent, mask_ratio=mask_ratio
)

print(f"Final Test Error (Silent Genes): {test_error_silent}")
print(f"Final Test Error (Non-Silent Genes): {test_error_non_silent}")

# Save the trained models
torch.save(rbm_silent.state_dict(), "rbm_silent_test_val_batch_02mask.pth")
torch.save(rbm_non_silent.state_dict(), "rbm_non_silent_test_val_batch_02mask.pth")

# Save the error plots
plt.figure()
plt.plot(train_errors_silent, label="Silent Genes (Train Error)")
plt.plot(val_errors_silent, label="Silent Genes (Validation Error)")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Training and Validation Error (Silent Genes) over 100 Iterations")
plt.legend()
plt.savefig("train_val_error_over_100_iterations_silent.png")

plt.figure()
plt.plot(train_errors_non_silent, label="Non-Silent Genes (Train Error)")
plt.plot(val_errors_non_silent, label="Non-Silent Genes (Validation Error)")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Training and Validation Error (Non-Silent Genes) over 100 Iterations")
plt.legend()
plt.savefig("train_val_error_over_100_iterations_non_silent.png")

# Plot test errors
plt.figure()
plt.bar(
    ["Silent Genes", "Non-Silent Genes"], [test_error_silent, test_error_non_silent]
)
plt.ylabel("Test Error")
plt.title("Test Error for Silent and Non-Silent Genes")
plt.savefig("test_error_silent_vs_non_silent.png")
