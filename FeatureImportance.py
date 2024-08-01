import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split  # Add this import
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, classification_report

# Load the cleaned data
df = pd.read_csv('cleaned_stock_data2.csv')

# Drop rows that are completely blank
df.dropna(how='all', inplace=True)

# Function to fill missing values with sector-based mean
def fill_missing_with_sector_mean(df):
    # Replace inf and -inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the mean of the column grouped by sector
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df.groupby('sector')[column].transform(lambda x: x.fillna(x.mean()))
    return df

# Fill missing values with sector-based mean
df = fill_missing_with_sector_mean(df)

# Transform sector names to 'Other' if not 'Energy', 'Utilities', or 'Technology'
sectors_to_keep = ['Energy', 'Utilities', 'Technology']
df['sector'] = df['sector'].apply(lambda x: x if x in sectors_to_keep else 'Other')

# Separate features and target
X = df.drop('sector', axis=1)
y = df['sector']

# Handle missing values by imputing with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(class_weights))

# Train a RandomForestClassifier to get feature importance with class weights
forest = RandomForestClassifier(random_state=42, class_weight='balanced')
forest.fit(X_scaled, y_encoded)

# Get feature importance
importances = forest.feature_importances_
feature_names = X.columns

# Ensure the lengths match
if len(feature_names) != len(importances):
    raise ValueError("Mismatch between number of features and importances")

feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("Feature Importances:\n", feature_importance_df)

# Select the top N features
top_n = 10  # Adjust N based on feature importance results
top_features = feature_importance_df.head(top_n)['feature']

# Update the preprocessing script to keep only the top features
X = df[top_features]
y = df['sector']

# Encode the target variable
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert encoded labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, class_weight=class_weights)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Get the original category names
target_names = list(map(str, label_encoder.classes_))

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=0))
