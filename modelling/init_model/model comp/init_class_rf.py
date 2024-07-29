### SCRIPT TO CLASSIFY GENES AS SILENT and NON-SILENT
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier


def init_classify_genes(threshold, undersample=False):
    # load data prepared for the model
    gene_matrix_array = np.load("gene_matrix_list.npy")
    rna_expression_df = pd.read_csv("rna_expression_list.csv")

    # check order of genes in both files is same
    assert gene_matrix_array.shape[0] == len(
        rna_expression_df
    ), "Mismatch in number of genes"

    # separate modification types
    dnam_features = gene_matrix_array[:, :, 0]  # Shape (58780, 4000)
    h3k9me3_features = gene_matrix_array[:, :, 1]
    h3k27me3_features = gene_matrix_array[:, :, 2]

    # concat features to make 1D
    X = np.concatenate(
        (dnam_features, h3k9me3_features, h3k27me3_features), axis=1
    )  # Shape (58780, 12000)

    #  binary target variable - classify if gene is silent or not
    y = (rna_expression_df["expression"].values > threshold).astype(
        int
    )  # 0 if silent, 1 if expressed

    if undersample:
        # Create a DataFrame to keep features and labels together
        data = pd.DataFrame(X)
        data["label"] = y

        # Split into silent and expressed classes
        silent = data[data["label"] == 0]
        expressed = data[data["label"] == 1]

        # Undersample the silent class
        silent_sampled = silent.sample(n=len(expressed), random_state=42)

        # Concatenate the undersampled silent class with the expressed class
        undersampled_data = pd.concat([silent_sampled, expressed])

        # Sort to maintain the original order
        undersampled_data = undersampled_data.sort_index()

        # Separate features and labels
        X = undersampled_data.drop("label", axis=1).values
        y = undersampled_data["label"].values

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # store eval metrics
    metrics = {}

    # train and eval
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Storing metrics
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred)
    metrics["Recall"] = recall_score(y_test, y_pred)
    metrics["F1-score"] = f1_score(y_test, y_pred)
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
    metrics["ConfusionMatrix"] = confusion_matrix(y_test, y_pred).tolist()

    print("Random Forest metrics:")
    print(metrics)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_class_rf.py <threshold> [undersample]")
        sys.exit(1)

    threshold = int(sys.argv[1])
    undersample = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    init_classify_genes(threshold, undersample)
