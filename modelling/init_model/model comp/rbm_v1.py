import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### DATA PREPARATION
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# check order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# Separate modification types
dnam_features = gene_matrix_array[:, :, 0]
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]

# concat features to make 1D
X = np.concatenate((dnam_features, h3k9me3_features, h3k27me3_features), axis=1)

# Separate expression values
expression_values = rna_expression_df["expression"].values

# separate data into silent and non-silent based on expression values
silent_indices = np.where(expression_values == 0)[0]  # Indices of silent genes
non_silent_indices = np.where(expression_values > 0)[0]  # Indices of non-silent genes

X_silent = X[silent_indices]  # Silent gene data
X_non_silent = X[non_silent_indices]  # Non-silent gene data

# Convert data to PyTorch tensors
X_silent = torch.tensor(X[silent_indices], dtype=torch.float32)  # Silent gene data
X_non_silent = torch.tensor(
    X[non_silent_indices], dtype=torch.float32
)  # Non-silent gene data


### RBM CLASS DEFINITION
class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        # two main layers
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weights and biases for hidden and visable layers
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

    def forward(self, v):
        # forward pass consists of calculating hidden proba then passing back to visible for v proba
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


# Function to train RBM and track metrics
def train(rbm, data, n_iter, lr=0.1, k=1):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    reconstruction_errors = []

    for i in range(n_iter):
        v0, v_prob, h_sample = rbm.contrastive_divergence(data, k)

        # Compute reconstruction error
        error = rbm.reconstruction_error(data, v_prob)
        reconstruction_errors.append(error.item())

        # Update weights
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

        print(f"Iteration {i + 1}/{n_iter}, Reconstruction Error: {error.item()}")

    return reconstruction_errors


# Train RBMs and track reconstruction errors
n_iterations = 50
reconstruction_errors_silent = train(rbm_silent, X_silent, n_iterations)
reconstruction_errors_non_silent = train(rbm_non_silent, X_non_silent, n_iterations)

# Save the trained models
torch.save(rbm_silent.state_dict(), "rbm_silent.pth")
torch.save(rbm_non_silent.state_dict(), "rbm_non_silent.pth")

# Save the reconstruction error plot
plt.plot(reconstruction_errors_silent, label="Silent Genes")
plt.plot(reconstruction_errors_non_silent, label="Non-Silent Genes")
plt.xlabel("Iterations")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error over Iterations")
plt.legend()
plt.savefig("Reconstruction_Error_over_Iterations.png")

# Print the final reconstruction error
print(f"Final Reconstruction Error (Silent Genes): {reconstruction_errors_silent[-1]}")
print(
    f"Final Reconstruction Error (Non-Silent Genes): {reconstruction_errors_non_silent[-1]}"
)
