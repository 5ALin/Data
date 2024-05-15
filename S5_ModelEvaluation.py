import torch
import pandas as pd
from share import create_folder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Load the training and validation sets
output_folder = r'C:\\Users\\samue\\Downloads\\newData\\new'
X_test = torch.load(f"{output_folder}\\X_test.csv")
y_test = torch.load(f"{output_folder}\\y_test.csv")

# Convert to PyTorch tensors
X_test_tensor = X_test.float()
y_test_tensor = y_test.float().view(-1, 1)

# Load the trained PyTorch model from .pth file
model_folder = r'C:\\Users\\samue\\Downloads\\newData\\trained_models'
model_path = os.path.join(model_folder, 'pytorch_churn_model.pth')

# Instantiate your PyTorch model
class ChurnPredictor(torch.nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load the PyTorch model
loaded_model = ChurnPredictor(input_size=X_test_tensor.shape[1])
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()

# Make predictions on the test set
with torch.no_grad():
    y_pred_tensor = loaded_model(X_test_tensor)

# Convert predictions to binary values (assuming threshold of 0.5)
y_pred_binary = (y_pred_tensor > 0.5).int()

# Calculate accuracy
correct_predictions = torch.sum(y_pred_binary == y_test_tensor.int())
total_samples = len(y_test_tensor)
accuracy = correct_predictions.item() / total_samples

# Create a confusion matrix
conf_matrix = torch.zeros(2, 2, dtype=torch.int32)
conf_matrix[0, 0] = torch.sum((y_test_tensor == 0) & (y_pred_binary == 0))
conf_matrix[0, 1] = torch.sum((y_test_tensor == 0) & (y_pred_binary == 1))
conf_matrix[1, 0] = torch.sum((y_test_tensor == 1) & (y_pred_binary == 0))
conf_matrix[1, 1] = torch.sum((y_test_tensor == 1) & (y_pred_binary == 1))

# Print the results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix.numpy()}')

# Analyze the model's predictions
prediction_analysis = pd.DataFrame({
    'Actual': np.ravel(y_test_tensor.numpy()),
    'Predicted': np.ravel(y_pred_binary.numpy())
})

# Create a folder for saving graphs within the Output folder
results_folder = r'C:\\Users\\samue\\Downloads\\newData\\output'
ci_graphs_folder = os.path.join(results_folder, 'ME_Graphs')
create_folder(ci_graphs_folder)

# Display the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix.numpy(), index=['0', '1'], columns=['0', '1'])
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(ci_graphs_folder, 'confusion_matrix.png'))
plt.show()
