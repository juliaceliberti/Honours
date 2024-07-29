### RF regression to predict methylation count (across 4000 bases) on K9, K27 and gene expression counts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


def predict_dna_methylation():

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
    X_histone = np.concatenate(
        (h3k9me3_features, h3k27me3_features), axis=1
    )  # Shape (58780, 8000)
    X_expression = rna_expression_df["expression"].values.reshape(
        -1, 1
    )  # Shape (58780, 1)
    X = np.concatenate((X_histone, X_expression), axis=1)  # Shape (58780, 8001)

    # Set DNA methylation as target - altering y to become a count of 1s in each 4000 length array
    y = np.sum(dnam_features, axis=1)  # Shape (58780,)

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and eval model
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    # Calc and print metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regressor metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)

    # Plot actual vs predicted counts
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
    plt.xlabel("Actual Counts")
    plt.ylabel("Predicted Counts")
    plt.title("XGB Actual vs Predicted DNA Methylation Counts")

    # Save the plot
    plt.savefig("xgb_actual_vs_predicted_dnam_counts.png")


if __name__ == "__main__":
    predict_dna_methylation()
