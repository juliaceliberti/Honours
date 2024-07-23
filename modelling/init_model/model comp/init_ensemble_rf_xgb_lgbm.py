## ensemble model to classify genes as silent (0) or non-silent (1)
# using random forest, xgboost and lightgbm

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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import sys


def classify_genes(threshold):
    # Load data prepared for the model
    gene_matrix_array = np.load("gene_matrix_list.npy")
    rna_expression_df = pd.read_csv("rna_expression_list.csv")

    # Check order of genes in both files is same
    assert gene_matrix_array.shape[0] == len(
        rna_expression_df
    ), "Mismatch in number of genes"

    # Separate modification types
    dnam_features = gene_matrix_array[:, :, 0]  # Shape (58780, 4000)
    h3k9me3_features = gene_matrix_array[:, :, 1]
    h3k27me3_features = gene_matrix_array[:, :, 2]

    # Concat features to make 1D
    X = np.concatenate(
        (dnam_features, h3k9me3_features, h3k27me3_features), axis=1
    )  # Shape (58780, 12000)

    # Binary target variable - classify if gene is silent or not
    y = (rna_expression_df["expression"].values > threshold).astype(
        int
    )  # 0 if silent, 1 if expressed

    # Training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize classifiers
    rf = RandomForestClassifier(random_state=42)
    lgbm = LGBMClassifier(random_state=42)
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")

    # Create a voting classifier
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lgbm", lgbm), ("xgb", xgb)],
        voting="soft",  # Use soft voting to average the probabilities
    )

    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    # Store evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print("Ensemble model metrics:")
    print(metrics)


if __name__ == "__main__":
    threshold = float(sys.argv[1])
    classify_genes(threshold)
