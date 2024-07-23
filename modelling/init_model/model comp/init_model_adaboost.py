### SCRIPT TO RUN INITIAL adaboost

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

### AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
mse_ada = mean_squared_error(y_test, y_pred_ada)
print(f"AdaBoost Mean Squared Error: {mse_ada}")
