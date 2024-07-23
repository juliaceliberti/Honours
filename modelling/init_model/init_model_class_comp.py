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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
y = (rna_expression_df["expression"].values > 0).astype(
    int
)  # 0 if silent, 1 if expressed

# training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# init classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="logloss"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "Support Vector Classifier": SVC(probability=True),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

# store eval metrics
metrics = {
    "Accuracy": {},
    "Precision": {},
    "Recall": {},
    "F1-score": {},
    "ROC-AUC": {},
    "ConfusionMatrix": {},
}

# train and eval
for clf_name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics["Accuracy"][clf_name] = accuracy_score(y_test, y_pred)
    metrics["Precision"][clf_name] = precision_score(y_test, y_pred)
    metrics["Recall"][clf_name] = recall_score(y_test, y_pred)
    metrics["F1-score"][clf_name] = f1_score(y_test, y_pred)
    metrics["ROC-AUC"][clf_name] = roc_auc_score(y_test, y_prob)
    metrics["ConfusionMatrix"][clf_name] = confusion_matrix(y_test, y_pred)

# results table
print("| Model Type  | Accuracy | Precision | Recall | F1-score | ROC-AUC |")
print(
    "| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |"
)
for clf_name in classifiers.keys():
    print(
        f"| {clf_name}  | {metrics['Accuracy'][clf_name]:.4f} | {metrics['Precision'][clf_name]:.4f} | {metrics['Recall'][clf_name]:.4f} | {metrics['F1-score'][clf_name]:.4f} | {metrics['ROC-AUC'][clf_name]:.4f} |"
    )

# confusion matrices
for clf_name, cm in metrics["ConfusionMatrix"].items():
    print(f"\nConfusion Matrix for {clf_name}:\n{cm}")
