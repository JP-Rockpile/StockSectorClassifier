import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the cleaned data
df = pd.read_csv('cleaned_stock_data2.csv')

# Separate features and target
X = df.drop('sector', axis=1)
y = df['sector']

# One-hot encode the target variable for processing but keep the original for final output
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

# Define numerical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns

# Define preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply preprocessing pipeline
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Convert numpy arrays back to DataFrames for saving to CSV
X_train_df = pd.DataFrame(X_train, columns=numerical_features)
X_test_df = pd.DataFrame(X_test, columns=numerical_features)
y_train_df = pd.DataFrame(y_train, columns=encoder.categories_[0])
y_test_df = pd.DataFrame(y_test, columns=encoder.categories_[0])

# Keep the original sector column in the final DataFrames
train_sectors = y[:len(X_train)]
test_sectors = y[len(X_train):]

# Concatenate features and labels for training and testing sets
train_df = pd.concat([X_train_df, train_sectors.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test_df, test_sectors.reset_index(drop=True)], axis=1)

# Save the processed training and testing data to CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("Processed data saved to train_data.csv and test_data.csv.")
