import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

# select histone modification for target value
k9 = True

# Load data
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# Check order of genes in both files is same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# Separate mod types
h3k9me3_features = gene_matrix_array[:, :, 1]  # Shape (58780, 4000)
h3k27me3_features = gene_matrix_array[:, :, 2]
dnam_features = gene_matrix_array[:, :, 0]  # Shape (58780, 4000)

# Concat histone mods and gene expression to create features
rna_expression = (rna_expression_df["expression"].values > 0).astype(
    int
)  # convert to binary
X_expression = rna_expression.reshape(-1, 1)  # Shape (58780, 1)

if k9:
    X = np.concatenate(
        (h3k27me3_features, dnam_features, X_expression), axis=1
    )  # Shape (58780, 8001)
    y = np.sum(h3k9me3_features, axis=1)  # Shape (58780,)
else:
    X = np.concatenate(
        (h3k9me3_features, dnam_features, X_expression), axis=1
    )  # Shape (58780, 8001)
    y = np.sum(h3k27me3_features, axis=1)  # Shape (58780,)

# Filter out the zero counts
non_zero_indices = np.where(y != 0)[0]

# Create a dataset with only non-zero counts
X_non_zero = X[non_zero_indices]
y_non_zero = y[non_zero_indices]

# Split the non-zero data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_non_zero, y_non_zero, test_size=0.2, random_state=42
)

# Train and evaluate the model
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

# Calculate and print metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("LGBM Regressor metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

histone = "K9" if k9 else "K27"

# Plot actual vs predicted counts
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
plt.xlabel("Actual Counts")
plt.ylabel("Predicted Counts")
plt.title("LGBM Actual vs Predicted {} Counts (Non-Zero)".format(histone))

# Save the plot
plt.savefig("lgbm_actual_vs_predicted_{}_counts_nonzero.png".format(histone))
