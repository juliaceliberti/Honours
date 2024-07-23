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

def classify_genes(threshold):
    # Load data prepared for the model
    gene_matrix_array = np.load("gene_matrix_list.npy")
    rna_expression_df = pd.read_csv("rna_expression_list.csv")

    # Check order of genes in both files is the same
    assert gene_matrix_array.shape[0] == len(rna_expression_df), "Mismatch in number of genes"

    # Separate modification types
    dnam_features = gene_matrix_array[:, :, 0]
    h3k9me3_features = gene_matrix_array[:, :, 1]
    h3k27me3_features = gene_matrix_array[:, :, 2]

    # Apply the threshold
    y = (rna_expression_df["expression"].values > threshold).astype(int)

    # Concatenate features to make 1D
    X = np.concatenate((dnam_features, h3k9me3_features, h3k27me3_features), axis=1)

    # Training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        print("Usage: python thr_135_class_xgboost.py <threshold>")
        sys.exit(1)
    threshold = int(sys.argv[1])
    classify_genes(threshold)
