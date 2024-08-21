import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
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

### RBM PARAMETERS
hiddenUnits = 256
visibleUnits = X_train.shape[1]  # This should match the number of features

# TensorFlow 2.x uses tf.Variable instead of placeholders
vb = tf.Variable(tf.zeros([visibleUnits]), dtype=tf.float32)
hb = tf.Variable(tf.zeros([hiddenUnits]), dtype=tf.float32)
W = tf.Variable(
    tf.random.normal([visibleUnits, hiddenUnits], mean=0.0, stddev=0.01),
    dtype=tf.float32,
)

# Input placeholder
v0 = tf.Variable(X_train, dtype=tf.float32)

# Phase 1: Input Processing
h0_prob = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
h0 = tf.nn.relu(tf.sign(h0_prob - tf.random.uniform(tf.shape(h0_prob))))

# Phase 2: Reconstruction
v1_prob = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
v1 = tf.nn.relu(tf.sign(v1_prob - tf.random.uniform(tf.shape(v1_prob))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

# Learning rate
alpha = 0.001

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)
CD = (w_pos_grad - w_neg_grad) / tf.cast(tf.shape(v0)[0], tf.float32)

# Update weights and biases using TensorFlow operations directly
print(f"Before update: W[0,0] = {W[0,0].numpy()}")
update_w = W.assign_add(alpha * CD)
print(f"After update: W[0,0] = {W[0,0].numpy()}")
update_vb = vb.assign_add(alpha * tf.reduce_mean(v0 - v1, axis=0))
update_hb = hb.assign_add(alpha * tf.reduce_mean(h0 - h1, axis=0))

# Error computation
err = v0 - v1
err_sum = tf.reduce_mean(err * err)

# Training the RBM
epochs = 15
batchsize = 100
errors = []

for epoch in range(epochs):
    for start in range(0, len(X_train), batchsize):
        end = start + batchsize
        batch = X_train[start:end]

        # Perform the updates and calculate error
        _, _, _, err_ = update_w, update_vb, update_hb, err_sum
        errors.append(err_)
        print(f"Epoch {epoch + 1}, Batch {start // batchsize + 1}: Error = {err_}")

# Plotting the error over epochs
plt.plot(errors)
plt.ylabel("Error")
plt.xlabel("Epoch")
plt.show()
