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
from sklearn.ensemble import RandomForestClassifier


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

thr1, thr3, thr5 = 1, 3, 5
thr1_y = (rna_expression_df["expression"].values > thr1).astype(
    int
)  # 0 if silent (1), 1 if expressed
thr3_y = (rna_expression_df["expression"].values > thr3).astype(
    int
)  # 0 if silent (3), 1 if expressed
thr5_y = (rna_expression_df["expression"].values > thr5).astype(
    int
)  # 0 if silent (5), 1 if expressed


# concat features to make 1D
X = np.concatenate(
    (dnam_features, h3k9me3_features, h3k27me3_features), axis=1
)  # Shape (58780, 12000)


i = 0
thr_list = [thr1, thr3, thr5]


for thr in [thr1_y, thr3_y, thr5_y]:
    #  binary target variable - classify if gene is silent or not
    y = thr

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
    print("Threshold:", thr_list[i])
    print(metrics)
    print("--------------------------")
    i += 1
