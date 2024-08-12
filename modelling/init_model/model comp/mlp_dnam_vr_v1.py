# Multi-Layer Perceptron - 3 dense layers with drop out & MSE for vector regression to predict DNAm
# Vector regression

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Prepare the data
target = "dnam"

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
rna_expression = (rna_expression_df["expression"].values > 0).astype(int)

if target == "dnam":
    y = dnam_features
    X = np.concatenate(
        (h3k9me3_features, h3k27me3_features, rna_expression.reshape(-1, 1)), axis=1
    )  # 1D 8001 binary array

elif target == "k9":
    y = h3k9me3_features
    X = np.concatenate(
        (dnam_features, h3k27me3_features, rna_expression.reshape(-1, 1)), axis=1
    )  # 1D 8001 binary array

elif target == "k27":
    y = h3k27me3_features
    X = np.concatenate(
        (dnam_features, h3k9me3_features, rna_expression.reshape(-1, 1)), axis=1
    )  # 1D 8001 binary array

# Training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tf.random.set_seed(42)

# Model params
input_size = 8001  # expression (1) + K9 (4000) + K27 (4000)
output_size = 4000  # predict DNAm at each position (4000)
batch_size = 32
epochs = 150

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=20,  # Stop after 5 epochs with no improvement
    restore_best_weights=True,  # Restore the best model weights
)

# Defining the MLP for Vector Regression
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            1000,
            input_shape=(input_size,),
            activation="relu",
            kernel_initializer="he_normal",
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(500, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(
            output_size, activation="linear"
        ),  # Output layer for vector regression
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss="mean_squared_error",  # Using MSE for regression
    metrics=["mae"],  # Mean Absolute Error as an additional metric
)

# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    verbose=2,
    callbacks=[early_stopping],
)

# Post-process the predictions to classify as 0 or 1
predictions = model.predict(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Evaluate the model with classification metrics
accuracy = accuracy_score(y_test, binary_predictions)
precision = precision_score(y_test, binary_predictions, average="micro")
recall = recall_score(y_test, binary_predictions, average="micro")
f1 = f1_score(y_test, binary_predictions, average="micro")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
