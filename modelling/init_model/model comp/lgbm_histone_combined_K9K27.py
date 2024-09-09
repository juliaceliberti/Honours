### LGBM model to predict combines count of K9 and K27 (x/4000 + x/4000)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor

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
rna_expression = rna_expression_df["expression"].values.reshape(
    -1, 1
)  # Shape (58780, 1)

X = np.concatenate((dnam_features, rna_expression), axis=1)  # Shape (58780, 4001)

# Set DNA methylation as target - altering y to become a count of 1s in each 4000 length array
y1 = np.sum(h3k9me3_features, axis=1)  # Shape (58780,)
y2 = np.sum(h3k27me3_features, axis=1)  # Shape (58780,)
y = y1 + y2  # combined count of K9 and K27 modifications, shape (58780,)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and eval model
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

# Calc and print metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("LGBM Regressor metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)


# Plot actual vs predicted counts
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
plt.xlabel("Actual Counts")
plt.ylabel("Predicted Counts")
plt.title("LGBM Actual vs Predicted K9+K27 Counts")

# Save the plot
plt.savefig("lgbm_actual_vs_predicted_K9K27_counts.png")
