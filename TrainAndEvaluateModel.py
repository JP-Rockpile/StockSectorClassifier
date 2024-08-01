import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the preprocessed training and testing data
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Separate features and labels
X_train = train_df.drop(columns=['sector']).values
y_train = train_df['sector'].values
X_test = test_df.drop(columns=['sector']).values
y_test = test_df['sector'].values

print("Data loaded and split into features and labels.")

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
class_weights = dict(enumerate(class_weights))

# Convert encoded labels to one-hot vectors
y_train = to_categorical(y_train_encoded)
y_test = to_categorical(y_test_encoded)

print("Target variable encoded.")

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')
])

print("Model architecture defined.")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model compiled.")

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32, class_weight=class_weights)

print("Model training complete.")

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Get the original category names
target_names = list(map(str, label_encoder.classes_))

# Debugging prints
print("Target Names:", target_names)
print("Unique values in y_true:", set(y_true))
print("Unique values in y_pred_classes:", set(y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", conf_matrix)

# Classification report
print("Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=target_names, zero_division=0))
