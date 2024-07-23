### SCRIPT TO RUN INITIAL RF MODEL USING RANDOMISED SEARCH

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# load data prepared for model
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# check the order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# separate the modification types
dnam_features = gene_matrix_array[:, :, 0]  # gets shape (58780, 4000)
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]

# concat the features along the feature dimension
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features), axis=1
)  # gets shape (58780, 12000)

# target
y = rna_expression_df["expression"].values  # gets shape (58780,)

# split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# init RF and use gridsearch
rf = RandomForestRegressor(random_state=42)


# fit model using random search outcomes
rf.fit(X_train, y_train)


# pred on test set
y_pred = rf.predict(X_test)

# eval model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# get feature importances (aggregated by feature type)
importances = rf.feature_importances_
dnam_importance = np.sum(importances[:4000])
h3k9me3_importance = np.sum(importances[4000:8000])
h3k27me3_importance = np.sum(importances[8000:])

print("Aggregated feature importances:")
print(f"DNAm: {dnam_importance}")
print(f"H3K9me3: {h3k9me3_importance}")
print(f"H3K27me3: {h3k27me3_importance}")
