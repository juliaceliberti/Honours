import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert training data to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# prepare batches for training
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the model
class MultiLayerNN(nn.Module):
    def __init__(
        self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes
    ):
        super(MultiLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout = nn.Dropout(p=0.5)  # Add drop out for regularsation
        self.relu1 = nn.ReLU()  # ReLU activation after 1st layer

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation after 2nd layer

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # third fc layer
        self.relu3 = nn.ReLU()  # ReLU activation after 2nd layer

        self.fc4 = nn.Linear(hidden_size3, 1)  # single output for BCE

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


# Define the model parameters
input_size = 4000 * 3  # Input size (4000 base pairs * 3 features)
hidden_size1 = 1000  # first hidden layer
hidden_size2 = 500  # second hidden layer
hidden_size3 = 500  # third hidden layer

num_classes = 2  # Number of output classes (binary classification)

# init model
model = MultiLayerNN(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)

# Define loss function and optimizer
criterion = (
    nn.BCEWithLogitsLoss()
)  # using BCE as data is binary and is combined with a sigmoid layer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()  # Squeeze outputs to match label shape
        loss = criterion(outputs, labels.float())  # Convert labels float
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # convert outputs to binary predictions
        predicted = (outputs > 0.5).int()
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

    # calc training metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    train_accuracy = accuracy_score(all_labels, all_predictions)
    train_precision = precision_score(all_labels, all_predictions)
    train_recall = recall_score(all_labels, all_predictions)
    train_f1 = f1_score(all_labels, all_predictions)

    print(
        f"Epoch {epoch+1}, Loss: {running_loss}, "
        f"Training Accuracy: {train_accuracy}, "
        f"Training Precision: {train_precision}, "
        f"Training Recall: {train_recall}, "
        f"Training F1 Score: {train_f1}"
    )

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test).squeeze()
    predicted = (outputs > 0.5).int()  # threshold for binary class
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
