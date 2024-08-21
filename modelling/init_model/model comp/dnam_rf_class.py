import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.metrics import confusion_matrix

# prep data - undersample not applicable for DNAm
undersample = False

# load data prepared for the model
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# check order of genes in both files is the same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# separate modification types
dnam_features = gene_matrix_array[:, :, 0]
h3k9me3_features = gene_matrix_array[:, :, 1]
h3k27me3_features = gene_matrix_array[:, :, 2]

# apply threshold
rna_expression = (rna_expression_df["expression"].values > 0).astype(int)

# concat features to make 1D (for X: histone mods and expression)
X = np.concatenate(
    (h3k9me3_features, h3k27me3_features, rna_expression.reshape(-1, 1)), axis=1
)
y = dnam_features  # y: DNAm

if undersample:
    # create a DataFrame to keep features and labels together
    data = pd.DataFrame(X)
    data["label"] = y

    # silent and expressed classes
    silent = data[data["label"] == 0]
    expressed = data[data["label"] == 1]

    # undersample the silent class
    silent_sampled = silent.sample(n=len(expressed), random_state=42)

    # concat the undersampled silent class with the expressed class
    undersampled_data = pd.concat([silent_sampled, expressed])

    # sort to maintain the original order
    undersampled_data = undersampled_data.sort_index()

    # separate features and labels
    X = undersampled_data.drop("label", axis=1).values
    y = undersampled_data["label"].values

# train and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# init the RandomForestClassifier with warm_start=True to avoid OOM kill event
# solution found here: https://stats.stackexchange.com/questions/327335/batch-learning-w-random-forest-sklearn
rf_classifier = RandomForestClassifier(
    n_estimators=10,  # start small number of trees and increase throughout training
    warm_start=True,
    random_state=42,
    n_jobs=-1,
)

batch_size = 10000
n_batches = int(np.ceil(X_train.shape[0] / batch_size))

# train model in batches
for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, X_train.shape[0])
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]

    # increase trees n_estimators for each new batch
    rf_classifier.n_estimators += 10  # 10 more trees with each batch

    rf_classifier.fit(X_batch, y_batch)

# Predict on the train set
y_train_pred = rf_classifier.predict(X_train)

# predict on the test set
y_test_pred = rf_classifier.predict(X_test)


# Evaluate the model on the validation set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred, average="macro")
train_recall = recall_score(y_train, y_train_pred, average="macro")
train_f1 = f1_score(y_train, y_train_pred, average="macro")

print(f"Validation Accuracy: {train_accuracy}")
print(f"Validation Precision: {train_precision}")
print(f"Validation Recall: {train_recall}")
print(f"Validation F1 Score: {train_f1}")

# Evaluate the model on the test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average="macro")
test_recall = recall_score(y_test, y_test_pred, average="macro")
test_f1 = f1_score(y_test, y_test_pred, average="macro")

print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test F1 Score: {test_f1}")


conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_test_pred.argmax(axis=1))
print("Confusion Matrix:")
print(conf_matrix)
