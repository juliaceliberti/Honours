import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load data
gene_matrix_array = np.load("gene_matrix_list.npy")
rna_expression_df = pd.read_csv("rna_expression_list.csv")

# Check order of genes in both files is same
assert gene_matrix_array.shape[0] == len(
    rna_expression_df
), "Mismatch in number of genes"

# Separate mod types
h3k9me3_features = gene_matrix_array[:, :, 1]  # Shape (58780, 4000)
h3k27me3_features = gene_matrix_array[:, :, 2]
dnam_features = gene_matrix_array[:, :, 0]  # Shape (58780, 4000)

# Concat histone mods and gene expression to create features
X_histone = np.concatenate(
    (h3k9me3_features, h3k27me3_features), axis=1
)  # Shape (58780, 8000)

### SWITCHING TO EXP COUNT
# rna_expression = (rna_expression_df["expression"].values > 0).astype(int)
# X_expression = rna_expression.reshape(-1, 1)  # Shape (58780, 1)

X_expression = rna_expression_df["expression"].values.reshape(-1, 1)  # Shape (58780, 1)


X = np.concatenate((X_histone, X_expression), axis=1)  # Shape (58780, 8001)

# Set DNA methylation as target - altering y to become a count of 1s in each 4000 length array
y = np.sum(dnam_features, axis=1)  # Shape (58780,)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the MLP model for regression
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            512, input_shape=(8001,), activation="relu", kernel_initializer="he_normal"
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.01),
        ),
        tf.keras.layers.Dense(1, activation="linear"),  # Output layer for regression
    ]
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="mean_squared_error",
    metrics=["mean_squared_error"],
)

# Train the model
batch_size = 32
epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=2,
)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse}")

# Plot training and validation loss
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("rf_loss_classification_dnam_counts.png")
