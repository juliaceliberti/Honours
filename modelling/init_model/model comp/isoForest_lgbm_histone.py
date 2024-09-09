### RF regression to predict Histone count (across 4000 bases) on K9 or K27, using the other histone mod (not set to target), gene expression and DNAm as features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

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

    # Set DNA methylation as target - altering y to become a count of 1s in each 4000 length array
    y = np.sum(h3k9me3_features, axis=1)  # Shape (58780,)
else:
    X = np.concatenate(
        (h3k9me3_features, dnam_features, X_expression), axis=1
    )  # Shape (58780, 8001)

    # Set DNA methylation as target - altering y to become a count of 1s in each 4000 length array
    y = np.sum(h3k27me3_features, axis=1)  # Shape (58780,)

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

### MODEL
from sklearn.ensemble import IsolationForest
import lightgbm as lgb
import numpy as np

# get only the zero-count samples (majority class)
X_train_majority = X_train[y_train == 0]

# init and train the isolation Forest
iso_forest = IsolationForest(contamination="auto", random_state=42)
iso_forest.fit(X_train_majority)

# detect anomalies (non-zero count samples) on the full training set
anomaly_scores_train = iso_forest.decision_function(X_train)
anomalies_train = iso_forest.predict(X_train)  # -1 for anomaly, 1 for normal

# filter out the anomalies (i.e., potential non-zero counts)
anomaly_indices_train = anomalies_train == -1
X_anomalies_train = X_train[anomaly_indices_train]
y_anomalies_train = y_train[anomaly_indices_train]

# init and train the LGBM Regressor
lgbm_regressor = lgb.LGBMRegressor(random_state=42)
lgbm_regressor.fit(X_anomalies_train, y_anomalies_train)

# predict anomalies on test set
anomalies_test = iso_forest.predict(X_test)  # -1 for anomaly & 1 for normal
anomaly_indices_test = anomalies_test == -1

# init the prediction array with zeros (for normal cases)
final_predictions = np.zeros_like(y_test)

# predict counts using the LGBM Regressor for the detected anomalies
final_predictions[anomaly_indices_test] = lgbm_regressor.predict(
    X_test[anomaly_indices_test]
)

# eval model performance
mse = mean_squared_error(y_test, final_predictions)
mae = mean_absolute_error(y_test, final_predictions)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

print("Iso Forest + LGBM metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

histone = "K9" if k9 else "K27"

# Plot actual vs predicted counts
plt.figure(figsize=(10, 6))
plt.scatter(y_test, final_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
plt.xlabel("Actual Counts")
plt.ylabel("Predicted Counts")
plt.title("IsoForest/LGBM Actual vs Predicted {} Counts".format(histone))

# Save the plot
plt.savefig("isoforest_lgbm_actual_vs_predicted_{}_counts.png".format(histone))
