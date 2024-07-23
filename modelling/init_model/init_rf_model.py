### SCRIPT TO RUN INITIAL RF MODEL USING RANDOMISED SEARCH

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import randint

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

# define params
param_dist = {
    "max_depth": [100, 500, 1000, None],
    "n_estimators": randint(50, 200),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
}

# init randomised search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
    n_jobs=-1,
)

# fit model using random search outcomes
random_search.fit(X_train, y_train)

# best parameters and the best model
best_params = random_search.best_params_
best_rf = random_search.best_estimator_

print(f"Best parameters: {best_params}")

# pred on test set
y_pred = best_rf.predict(X_test)

# eval model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# get feature importances (aggregated by feature type)
importances = best_rf.feature_importances_
dnam_importance = np.sum(importances[:4000])
h3k9me3_importance = np.sum(importances[4000:8000])
h3k27me3_importance = np.sum(importances[8000:])

print("Aggregated feature importances:")
print(f"DNAm: {dnam_importance}")
print(f"H3K9me3: {h3k9me3_importance}")
print(f"H3K27me3: {h3k27me3_importance}")
