import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from share import create_folder

# Load the cleaned data
file_path = 'C:\\Users\\samue\\Downloads\\newData\\output\\s2_output.csv'
df = pd.read_csv(file_path)

# Features (X) and Target Variable (y)
X = df.drop('churn', axis=1)
y = df['churn']

# Convert features and labels to PyTorch tensors
X_tensor = torch.FloatTensor(X.values)
y_tensor = torch.FloatTensor(y.values).view(-1, 1)  # Ensure it's a column vector

# Define your own method to split the data into training and testing sets
def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    # Implement your custom logic for splitting the data
    # For example, you can randomly shuffle the indices and split them

    # Corrected line: shuffle and assign back to indices
    indices = list(range(len(X)))
    torch.manual_seed(random_state)
    indices = torch.randperm(len(indices))
    split_index = int((1 - test_size) * len(indices))

    X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
    y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

    return X_train, X_test, y_train, y_test

# Specify the new output folder
output_folder = r'C:\\Users\\samue\\Downloads\\newData\\new'
create_folder(output_folder)

# Use your custom split function
X_train, X_test, y_train, y_test = custom_train_test_split(X_tensor, y_tensor)

# Save the training and testing sets to the new output folder
torch.save(X_train, f"{output_folder}\\X_train.csv")
torch.save(X_test, f"{output_folder}\\X_test.csv")
torch.save(y_train, f"{output_folder}\\y_train.csv")
torch.save(y_test, f"{output_folder}\\y_test.csv")
