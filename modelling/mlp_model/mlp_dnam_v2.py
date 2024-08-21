import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import class_weight
from keras import backend as K

# prep data
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

if undersample:  # not used in baseline
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
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)


tf.random.set_seed(42)

# optimiser
learning_rate = 0.001
momentum = 0.9

# using SGD to prevent overfitting with momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

# L1 and L2 regularization parameters
l1 = 0.01
l2 = 0.01

# model params
input_size = 8001
output_size = 4000

# defining the MLP layers with L1 and L2 regularization
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            6000,
            input_shape=(input_size,),
            activation="relu",
            kernel_initializer="he_normal",
            # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            5000,
            activation="relu",
            kernel_initializer="he_normal",
            # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
        ),
        tf.keras.layers.Dense(
            output_size,
            activation="sigmoid",
            # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2),
        ),
    ]
)


# custom weighted BCE for imbalanced positive and neg ouputs
def weighted_binary_crossentropy(y_true, y_pred):
    # weights for each label (where 0.025 represents the presence of positive classs in outputs)
    weight_for_0 = 1 / (1 - 0.4)  # lower weight for 0
    weight_for_1 = 1 / 0.4  # much higher weight for 1

    # get weights tensor
    weights = y_true * weight_for_1 + (1 - y_true) * weight_for_0

    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
    y_pred_logits = tf.math.log(y_pred / (1 - y_pred))

    # use pre-built binary cross-entropy loss
    bce = tf.nn.weighted_cross_entropy_with_logits(
        logits=y_pred_logits, labels=y_true, pos_weight=weight_for_1
    )

    # use weights to get loss
    weighted_bce = weights * bce

    return tf.reduce_mean(weighted_bce)


# Compile the model
model.compile(
    optimizer=optimizer,
    loss=weighted_binary_crossentropy,  # appropriate for multi-output binary classification
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

batch_size = 32
epochs = 150

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
)

# Train the model
history = model.fit(
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
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Final Accuracy on test set: {accuracy}")
print(f"Final Precision on test set: {precision}")
print(f"Final Recall on test set: {recall}")
print(f"Final F1 Score on test set: {f1}")


# Plotting the training and validation loss over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs (MLP predicting DNAm)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("mlp_dnam_error_v2.png")
