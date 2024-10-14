import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from kerastuner import RandomSearch

# Load and preprocess data (replace with your dataset path)
def load_data(file_path):
    # Example of loading a CSV file
    # Assuming the last column is the target and all previous columns are features
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values  # All columns except the last
    y = df.iloc[:, -1].values   # Last column as target
    
    # Reshape X to 3D (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Adjust based on your data shape
    return X, y

# Replace with your actual data file path
file_path = 'path/to/your/dataset.csv'
X, y = load_data(file_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model building function for Keras Tuner
def build_1d_cnn_model(hp):
    model = Sequential()

    # Hyperparameters for tuning
    hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
    hp_kernel_size = hp.Int('kernel_size', min_value=3, max_value=7, step=2)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)

    # Add layers to the model
    model.add(Conv1D(filters=hp_filters, kernel_size=hp_kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp_dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Initialize Keras Tuner
tuner = RandomSearch(
    build_1d_cnn_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='1d_cnn_tuning',
    project_name='cnn_1d'
)

# Search for the best hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Get the optimal hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
print(f"Best hyperparameters: {best_hp}")

# Build and train the best model
best_model = tuner.hypermodel.build(best_hp)
history = best_model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
