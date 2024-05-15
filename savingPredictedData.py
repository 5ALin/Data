# Example code to save predicted data to a CSV file
import pandas as pd
from joblib import load

# Load the trained model
model_folder = r'C:\\Users\\samue\\Downloads\\dataBiz\\trained_models'
clf = load(f"{model_folder}\\churn_model.joblib")

# Load the cleaned data
file_path = r'C:\\Users\\samue\\Downloads\\dataBiz\\output\\s2_output.csv'
df = pd.read_csv(file_path)

# Features (X) and Target Variable (y)
X = df.drop('churn', axis=1)

# Make predictions on the entire dataset
df['predicted_churn'] = clf.predict(X)

# Save the predicted data to a new CSV file
predicted_file_path = r'C:\\Users\\samue\\Downloads\\dataBiz\\output\\predicted_data.csv'
df.to_csv(predicted_file_path, index=False)
