import pandas as pd
from joblib import load
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the cleaned data
file_path = r'C:\\Users\\samue\\Downloads\\newData\\output\\s2_output.csv'  # Corrected path
df = pd.read_csv(file_path)

# Features (X) and Target Variable (y)
X = df.drop('churn', axis=1).values  # Convert DataFrame to NumPy array
y = df['churn'].values

# Instantiate the PyTorch model
class ChurnModel(nn.Module):
    def __init__(self, input_size):
        super(ChurnModel, self).__init__()
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

# Instantiate the PyTorch model
input_size = X.shape[1]  # Use the number of features as input size
pytorch_model = ChurnModel(input_size)

# Load the trained PyTorch model
model_folder = r'C:\\Users\\samue\\Downloads\\newData\\trained_models'  # Corrected path
model_path = os.path.join(model_folder, 'pytorch_churn_model.pth')  # Corrected file extension
pytorch_model.load_state_dict(torch.load(model_path))
pytorch_model.eval()

# Convert X to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)  # Set requires_grad=True for calculating gradients

# Make predictions on the entire dataset using the PyTorch model
y_pred_scores = pytorch_model(X_tensor)

# Create a tensor of ones with the same shape as y_pred_scores
ones_tensor = torch.ones_like(y_pred_scores, dtype=torch.float32)

# Calculate gradients
X_tensor.grad = None  # Clear gradients before backward pass
y_pred_scores.backward(ones_tensor, retain_graph=True)

# Calculate feature importances using gradients
gradients = X_tensor.grad.abs()
feature_importance = gradients.sum(dim=0)

# Normalize the feature importances
feature_importance /= feature_importance.sum()

# Convert predictions to binary (assuming binary classification)
y_pred = (y_pred_scores > 0.5).int().numpy()

# Features (X) and Target Variable (y) for classification metrics
y_true = df['churn'].values

# Calculate classification metrics using PyTorch
confusion_matrix = torch.zeros(2, 2)
for t, p in zip(y_true, y_pred):
    # Cast t and p to integers
    t, p = int(t), int(p)
    confusion_matrix[t, p] += 1

# Calculate precision, recall, and F1 score
precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
f1_score = 2 * (precision * recall) / (precision + recall)

# Display classification metrics
class_report = f"Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}"
print(f'Classification Metrics:\n{class_report}')

# Calculate gradients
y_pred_scores.backward(ones_tensor, retain_graph=True)

# Calculate feature importances using gradients
gradients = X_tensor.grad.abs()
feature_importance = gradients.sum(dim=0)

# Normalize the feature importances
feature_importance /= feature_importance.sum()

# Create a folder for saving graphs within the Output folder
output_folder = 'C:\\Users\\samue\\Downloads\\newData\\output'  # Corrected path
ci_graphs_folder = os.path.join(output_folder, 'CI_Graphs')
os.makedirs(ci_graphs_folder, exist_ok=True)

# Feature Importance Plot
plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importance.numpy(), y=df.drop('churn', axis=1).columns)
plt.title('Feature Importance')
plt.savefig(os.path.join(ci_graphs_folder, 'feature_importance.png'))
plt.show()