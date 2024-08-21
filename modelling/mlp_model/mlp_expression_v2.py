# Multi-Layer Perceptron - 3 dense layers with drop out & BCE to predict expression

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

# Training and testing
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


tf.random.set_seed(42)

# Define the learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True,
)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

# model params
input_size = 12000
output_size = 1

# defining the MLP layers - 3 dense layers with drop out & BCE to predict expression
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            20,
            input_shape=(input_size,),
            activation="relu",
            kernel_initializer="he_normal",
            # kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
        ),  # Using HeNormal for ReLU gradients
        tf.keras.layers.Dropout(0.5),  # add some regularisation
        tf.keras.layers.Dense(
            5,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.01),
        ),
        tf.keras.layers.Dense(
            output_size, activation="sigmoid"
        ),  # output layer for binary classification
    ]
)


# Compile the model
model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

batch_size = 32
epochs = 150

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=20,  # Stop after 5 epochs with no improvement
    restore_best_weights=True,  # Restore the best model weights
)


# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=[early_stopping],
)

# Calculate and print final metrics
y_pred = (model.predict(X_test) > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Final Accuracy on test set: {accuracy}")
print(f"Final Precision on test set: {precision}")
print(f"Final Recall on test set: {recall}")
print(f"Final F1 Score on test set: {f1}")
