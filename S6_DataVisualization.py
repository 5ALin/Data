import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the cleaned data
file_path = 'C:\\Users\\samue\\Downloads\\newData\\output\\s2_output.csv'
df = pd.read_csv(file_path)

# Create a folder for saving graphs within the Output folder
output_folder = 'C:\\Users\\samue\\Downloads\\newData\\output'
graphs_folder = os.path.join(output_folder, 'DV_Graphs')
os.makedirs(graphs_folder, exist_ok=True)

# Identify numeric columns
numeric_features = df.select_dtypes(include=['float32', 'int32', 'int64', 'float64']).columns

# Visualize the distribution of numeric features
plt.figure(figsize=(15, 10))
df[numeric_features].hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Numeric Features", y=1.02)
plt.savefig(os.path.join(graphs_folder, 'numeric_distribution.png'))
plt.show()

# Visualize correlations between numeric features
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Heatmap of Numeric Features")
plt.savefig(os.path.join(graphs_folder, 'correlation_heatmap.png'))
plt.show()

# Visualize the count of categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='churn', data=df)
    plt.title(f"Count of {feature} by Churn")
    plt.savefig(os.path.join(graphs_folder, f'{feature}_count.png'))
    plt.show()

# Scatter plots for relationships between numeric features
scatter_pairs = [('purchase_count', 'total_spend'), ('average_order_value', 'days_since_last_purchase')]
for pair in scatter_pairs:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pair[0], y=pair[1], hue='churn', data=df)
    plt.title(f'Scatter Plot of {pair[0]} vs {pair[1]}')
    plt.savefig(os.path.join(graphs_folder, f'{pair[0]}_{pair[1]}_scatter.png'))
    plt.show()
