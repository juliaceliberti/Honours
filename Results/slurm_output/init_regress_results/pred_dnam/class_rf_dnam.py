### RF multioutput classification to predict methylation at each base based on K9, K27 and gene expression counts

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import sys


## libraries for distributed computation
from dask.distributed import Client


def predict_dna_methylation(threshold, undersample=False):
    # init dask client

    client = Client()

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

    # Set DNA methylation as target
    y = dnam_features  # Shape (58780, 4000)

    #  undersample the majority class (if req)
    if undersample:
        # Create a DataFrame to keep features and labels together
        data = pd.DataFrame(X)
        data["label"] = (y.sum(axis=1) > threshold).astype(
            int
        )  # agg DNAm to get binary label

        # Split into silent and expressed classes
        silent = data[data["label"] == 0]
        expressed = data[data["label"] == 1]

        # Undersample silent class
        silent_sampled = silent.sample(n=len(expressed), random_state=42)

        # Concat the undersampled silent class with the expressed class
        undersampled_data = pd.concat([silent_sampled, expressed]).sort_index()

        # Separate features and labels
        X = undersampled_data.drop("label", axis=1).values
        undersampled_indices = undersampled_data.index
        y = y[undersampled_indices]

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and eval the model
    metrics = {}
    rf = RandomForestClassifier(random_state=42)
    multi_target_rf = MultiOutputClassifier(rf, n_jobs=-1)
    multi_target_rf.fit(X_train, y_train)
    y_pred = multi_target_rf.predict(X_test)

    # Storing metrics for each target
    metrics["Accuracy"] = [
        accuracy_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])
    ]
    metrics["Precision"] = [
        precision_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])
    ]
    metrics["Recall"] = [
        recall_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])
    ]
    metrics["F1-score"] = [
        f1_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])
    ]

    # Print average metrics
    print("Random Forest metrics:")
    print("Average Accuracy:", np.mean(metrics["Accuracy"]))
    print("Average Precision:", np.mean(metrics["Precision"]))
    print("Average Recall:", np.mean(metrics["Recall"]))
    print("Average F1-score:", np.mean(metrics["F1-score"]))

    # Close Dask client
    client.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_dna_methylation.py <threshold> [undersample]")
        sys.exit(1)

    threshold = int(sys.argv[1])
    undersample = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    predict_dna_methylation(threshold, undersample=False)
