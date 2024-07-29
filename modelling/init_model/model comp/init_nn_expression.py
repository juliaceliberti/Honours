import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Prepare the data
undersample = False

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert training data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define the model
class SingleLayerNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SingleLayerNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.batch_norm = nn.BatchNorm1d(input_size)  # Add batch normalization

    def forward(self, x):
        x = self.batch_norm(x)  # Apply batch normalization
        x = self.fc(x)
        return x


# Define the model parameters
input_size = 4000 * 3  # Input size (4000 base pairs * 3 features)
num_classes = 2  # Number of output classes (binary classification)

# Instantiate the model
model = SingleLayerNN(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    accuracy = 100 * correct / total

    y_test_np = y_test.numpy()
    predicted_np = predicted.numpy()
    precision = precision_score(y_test_np, predicted_np)
    recall = recall_score(y_test_np, predicted_np)
    f1 = f1_score(y_test_np, predicted_np)

print(f"Accuracy: {accuracy}%")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
