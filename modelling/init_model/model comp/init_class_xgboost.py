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
from xgboost import XGBClassifier


def init_classify_genes(threshold, undersample=False):
    # Load data prepared for the model
    gene_matrix_array = np.load("gene_matrix_list.npy")
    rna_expression_df = pd.read_csv("rna_expression_list.csv")

    # Check order of genes in both files is the same
    assert gene_matrix_array.shape[0] == len(
        rna_expression_df
    ), "Mismatch in number of genes"

    # Separate modification types
    dnam_features = gene_matrix_array[:, :, 0]
    h3k9me3_features = gene_matrix_array[:, :, 1]
    h3k27me3_features = gene_matrix_array[:, :, 2]

    # Apply the threshold
    y = (rna_expression_df["expression"].values > threshold).astype(int)

    # Concatenate features to make 1D
    X = np.concatenate((dnam_features, h3k9me3_features, h3k27me3_features), axis=1)

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

    # Training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Store eval metrics
    metrics = {}

    # Train and eval
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    # Storing metrics
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred)
    metrics["Recall"] = recall_score(y_test, y_pred)
    metrics["F1-score"] = f1_score(y_test, y_pred)
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
    metrics["ConfusionMatrix"] = confusion_matrix(y_test, y_pred).tolist()

    print("XGBoost metrics:")
    print("Threshold:", threshold)
    print(metrics)
    print("--------------------------")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python init_class_xgboost.py <threshold> [undersample]")
        sys.exit(1)

    threshold = int(sys.argv[1])
    undersample = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    init_classify_genes(threshold, undersample)
