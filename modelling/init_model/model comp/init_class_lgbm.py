### SCRIPT TO CLASSIFY GENES AS SILENT and NON-SILENT
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
import sys


def classify_genes(threshold):
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

    # training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # store eval metrics
    metrics = {}

    # train and eval
    lgbm = LGBMClassifier(random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)[:, 1]

    # Storing metrics
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred)
    metrics["Recall"] = recall_score(y_test, y_pred)
    metrics["F1-score"] = f1_score(y_test, y_pred)
    metrics["ROC-AUC"] = roc_auc_score(y_test, y_prob)
    metrics["ConfusionMatrix"] = confusion_matrix(y_test, y_pred).tolist()

    print("SVC metrics:")
    print(metrics)


if __name__ == "__main__":
    threshold = int(sys.argv[1])
    classify_genes(threshold)
