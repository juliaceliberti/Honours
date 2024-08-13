# # Restricted Boltzmann Model using all data (DNAm, K9, K27, expression)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

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
expression_values = rna_expression_df["expression"].values.reshape(
    -1, 1
)  # to make 2D like other features

# concat all features, including expression, to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
)

# convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


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
n_visible = X.shape[1]
n_hidden = 256

rbm = RBM(n_visible, n_hidden)


# Function to train RBM and track metrics
# Function to train RBM and track metrics
def train(rbm, dataloader, n_iter, lr=0.1, k=1):
    optimizer = optim.SGD(rbm.parameters(), lr=lr)
    reconstruction_errors = []

    for i in range(n_iter):
        batch_errors = []
        for batch in dataloader:
            data = batch[0]
            v0, v_prob, h_sample = rbm.contrastive_divergence(data, k)

            # Compute reconstruction error
            error = rbm.reconstruction_error(data, v_prob)
            batch_errors.append(error.item())  # Collect batch errors

            # Update weights
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        avg_error = np.mean(
            batch_errors
        )  # Calculate the average error for this iteration
        reconstruction_errors.append(avg_error)
        print(f"Iteration {i + 1}/{n_iter}, Reconstruction Error: {avg_error}")

    return reconstruction_errors


# Train RBMs and track reconstruction errors
n_iterations = 100
reconstruction_errors_all_data = train(rbm, dataloader, n_iterations)

# Save the trained models
torch.save(rbm.state_dict(), "rbm_all_data_v1.pth")

# Save the reconstruction error plot
plt.plot(reconstruction_errors_all_data, label="All Genes")
plt.xlabel("Iterations")
plt.ylabel("Reconstruction Error")
plt.title("Reconstruction Error over 100 Iterations")
plt.legend()
plt.savefig("reconstruction_error_over_100_iterations_all_data.png")

# Print the final reconstruction error
print(
    f"Final Reconstruction Error (Silent Genes): {reconstruction_errors_all_data[-1]}"
)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader, TensorDataset


# ### DATA PREPARATION
# gene_matrix_array = np.load("gene_matrix_list.npy")
# rna_expression_df = pd.read_csv("rna_expression_list.csv")

# # check order of genes in both files is the same
# assert gene_matrix_array.shape[0] == len(
#     rna_expression_df
# ), "Mismatch in number of genes"

# # separate modification types
# dnam_features = gene_matrix_array[:, :, 0]
# h3k9me3_features = gene_matrix_array[:, :, 1]
# h3k27me3_features = gene_matrix_array[:, :, 2]
# expression_values = rna_expression_df["expression"].values.reshape(
#     -1, 1
# )  # to make 2D like other features

# # concat all features, including expression, to make 1D
# X = np.concatenate(
#     (dnam_features, h3k9me3_features, h3k27me3_features, expression_values), axis=1
# )

# # convert data to PyTorch tensors
# X = torch.tensor(X, dtype=torch.float32)
# dataset = TensorDataset(X)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ### RBM CLASS DEFINITION
# class RBM(nn.Module):
#     def __init__(self, n_visible, n_hidden):
#         super(RBM, self).__init__()
#         # two main layers
#         self.n_visible = n_visible
#         self.n_hidden = n_hidden

#         # Weights and biases for hidden and visable layers
#         self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
#         self.h_bias = nn.Parameter(torch.zeros(n_hidden))
#         self.v_bias = nn.Parameter(torch.zeros(n_visible))

#     def sample_h(self, v):
#         # get hidden layer class proba
#         h_prob = torch.sigmoid(F.linear(v, self.W, self.h_bias))
#         return h_prob, torch.bernoulli(h_prob)

#     def sample_v(self, h):
#         # get visible layer class proba
#         v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
#         return v_prob, torch.bernoulli(v_prob)

#     def forward(self, v):
#         # forward pass consists of calculating hidden proba then passing back to visible for v proba
#         h_prob, h_sample = self.sample_h(v)
#         v_prob, v_sample = self.sample_v(h_sample)
#         return v_prob

#     def contrastive_divergence(self, v, k=1):
#         # training method
#         v0 = v
#         for _ in range(k):
#             h_prob, h_sample = self.sample_h(v0)
#             v_prob, v0 = self.sample_v(h_sample)
#         return v0, v_prob, h_sample

#     def reconstruction_error(self, v, v_reconstructed):
#         # calculate error
#         return torch.mean(torch.sum((v - v_reconstructed) ** 2, dim=1))


# # init RBM model
# n_visible = X.shape[1]  # 12001
# n_hidden = 256

# rbm = RBM(n_visible, n_hidden)


# # Function to train RBM and track metrics
# def train(rbm, dataloader, n_iter, lr=0.1, k=1):
#     optimizer = optim.SGD(rbm.parameters(), lr=lr)
#     reconstruction_errors = []

#     for i in range(n_iter):
#         batch_errors = []
#         for batch in dataloader:
#             data = batch[0]
#             v0, v_prob, h_sample = rbm.contrastive_divergence(data, k)

#             # Compute reconstruction error
#             error = rbm.reconstruction_error(data, v_prob)
#             reconstruction_errors.append(error.item())

#             # Update weights
#             optimizer.zero_grad()
#             error.backward()
#             optimizer.step()

#         avg_error = np.mean(batch_errors)
#         reconstruction_errors.append(avg_error)
#         print(f"Iteration {i + 1}/{n_iter}, Reconstruction Error: {avg_error}")

#     return reconstruction_errors


# # train RBM and track reconstruction errors
# n_iterations = 100
# reconstruction_errors = train(rbm, X, n_iterations)

# # Save model
# torch.save(rbm.state_dict(), "rbm_model_dnam_100.pth")

# # Save error plot
# plt.plot(reconstruction_errors, label="Reconstruction Error")
# plt.xlabel("Iterations")
# plt.ylabel("Reconstruction Error")
# plt.title("Reconstruction Error over 100 Iterations")
# plt.legend()
# plt.savefig("reconstruction_error_over_100_iterations_all_data_2.png")

# # print final error
# print(f"Final Reconstruction Error: {reconstruction_errors[-1]}")
