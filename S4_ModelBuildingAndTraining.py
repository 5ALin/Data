import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from share import create_folder, delete_file
import matplotlib.pyplot as plt

# Load the training and validation sets
output_folder = r'C:\\Users\\samue\\Downloads\\newData\\new'
X_train = torch.load(f"{output_folder}\\X_train.csv")
y_train = torch.load(f"{output_folder}\\y_train.csv")
X_val = torch.load(f"{output_folder}\\X_test.csv")
y_val = torch.load(f"{output_folder}\\y_test.csv")

# Convert to PyTorch tensors
X_train_tensor = X_train.float()
y_train_tensor = y_train.float().view(-1, 1)
X_val_tensor = X_val.float()
y_val_tensor = y_val.float().view(-1, 1)

# Create DataLoaders for training and validation
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class ChurnPredictor(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Train model with early stopping
model = ChurnPredictor(input_size=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 200
lowest_loss = float('inf')
patience = 10  # Number of epochs to wait if the validation loss does not improve
wait_count = 0

train_losses = []  # List to store training losses for each epoch
val_losses = []  # List to store validation losses for each epoch

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Calculate average training loss for the epoch
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_batch_x, val_batch_y in val_loader:
            val_output = model(val_batch_x)
            val_loss += criterion(val_output, val_batch_y).item()

    # Calculate average validation loss for the epoch
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Check if the current epoch has the lowest validation loss
    if val_loss < lowest_loss:
        lowest_loss = val_loss
        wait_count = 0
        # Save the PyTorch model at the epoch with the lowest validation loss
        model_folder = r'C:\\Users\\samue\\Downloads\\newData\\trained_models'
        create_folder(model_folder)
        delete_file(f"{model_folder}\\pytorch_churn_model.pth")
        torch.save(model.state_dict(), f"{model_folder}\\pytorch_churn_model.pth")
    else:
        wait_count += 1
        if wait_count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Plot the loss vs epochs
plt.plot(range(1, epoch + 2), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.plot(range(1, epoch + 2), val_losses, marker='o', linestyle='-', color='r', label='Validation Loss')
plt.title('Training and Validation Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot as a PNG
output_folder = r'C:\\Users\\samue\\Downloads\\newData\\output'
create_folder(output_folder)
plt.savefig(f"{output_folder}\\LossVsEpochs.png")

# Display the plot
plt.show()

# Load the PyTorch model
loaded_model = ChurnPredictor(input_size=X_train.shape[1])
loaded_model.load_state_dict(torch.load(f"{model_folder}\\pytorch_churn_model.pth"))
loaded_model.eval()
