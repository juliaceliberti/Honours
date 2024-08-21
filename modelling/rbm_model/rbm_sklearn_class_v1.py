import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prepare the data
undersample = True

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
y = (rna_expression_df["expression"].values > 0).astype(int)

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

# split train and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# RBM model
rbm = BernoulliRBM(n_components=500, learning_rate=0.01, n_iter=10, random_state=42)

# logistic classifier
logistic = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

# Create a pipeline that first applies RBM then Logistic Regression
classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# Train the model
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1 Score: {f1}")
