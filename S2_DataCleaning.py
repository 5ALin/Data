import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from share import create_folder


pd.set_option('display.max_columns', None)

# Load the df
file_path = 'C:\\Users\\samue\\Downloads\\newData\\output\\s1_output.csv'
df = pd.read_csv(file_path)

# Drop bad columns
print("Column Names:", df.columns.tolist())
df = df.drop(['customer_id', 'email', 'first_name', 'last_name'], axis=1)
print("Column Names:", df.columns.tolist())

# Convert 'registration_date' to datetime and extract relevant features
df['registration_date'] = pd.to_datetime(df['registration_date'])
df['registration_year'] = df['registration_date'].dt.year
df['registration_month'] = df['registration_date'].dt.month
df['registration_day'] = df['registration_date'].dt.day

# Drop the original 'registration_date' column
df = df.drop('registration_date', axis=1)
print("Column Names:", df.columns.tolist())

# Normalize string columns (convert to lowercase and strip whitespace)
columns_to_normalize = ['gender', 'city', 'state']
df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: x.str.lower().str.strip())
print(df.head(3))

# Normalize boolean columns
df['has_newsletter_subscription'] = df['has_newsletter_subscription'].astype(int)
df['has_returned_items'] = df['has_returned_items'].astype(int)
df['churn'] = df['churn'].astype(inwht)
print(df.head(3))

# Display column df types
print("Column Data Types:")
print(df.dtypes)

# Initialize LabelEncoders for city, state, gender
label_encoder_city = LabelEncoder()
label_encoder_state = LabelEncoder()
label_encoder_gender = LabelEncoder()

# Fit and transform city, state, gender
df['city_encoded'] = label_encoder_city.fit_transform(df['city'])
df['state_encoded'] = label_encoder_state.fit_transform(df['state'])
df['gender_encoded'] = label_encoder_gender.fit_transform(df['gender'])

# Save the LabelEncoders using joblib
label_encoder_folder = r'C:\\Users\\samue\\Downloads\\newData\\label_encoders'
create_folder(label_encoder_folder)
dump(label_encoder_city, f"{label_encoder_folder}\\label_encoder_city.joblib")
dump(label_encoder_state, f"{label_encoder_folder}\\label_encoder_state.joblib")
dump(label_encoder_gender, f"{label_encoder_folder}\\label_encoder_gender.joblib")

df = df.drop(['city', 'state', 'gender'], axis=1)
print("Column Names:", df.columns.tolist())
print(df.head(3))

# Convert all columns to float32
df = df.astype('float32')
print(df.dtypes)

# Save the dataset to a CSV file
folder_path = r'C:\\Users\\samue\\Downloads\\newData\\output'
create_folder(folder_path)
df.to_csv(f"{folder_path}\\s2_output.csv", index=False)