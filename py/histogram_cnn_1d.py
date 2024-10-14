import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Parameters
heatmaps_dir = 'path/to/AlgX3_64x64_merged_tensors_with_labels'  # Change this path
categories = ['0', '1']
img_size = 64
sequence_length = img_size * img_size  # 64 * 64
batch_size = 32
epochs = 50

# Initialize lists to hold image data and labels
images = []
labels = []

# Read images and labels
for category in categories:
    category_path = os.path.join(heatmaps_dir, category)
    for filename in os.listdir(category_path):
        if filename.endswith('.png'):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))

            img_array = img_to_array(img).flatten()
            images.append(img_array)
            labels.append(int(category))

# Convert lists to NumPy arrays
X = np.array(images) / 255.0  # Normalize
y = np.array(labels)

# Reshape for 1D CNN
X = X.reshape((X.shape[0], sequence_length, 1))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=16),
        kernel_size=hp.Int('kernel_size_1', min_value=3, max_value=7, step=2),
        activation='relu',
        input_shape=(sequence_length, 1)
    ))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=32),
        activation='relu'
    ))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Tuner for Bayesian optimization
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='bayesian_tuning',
    project_name='cnn_1d'
)

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform the search
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

# Get and evaluate the best model
best_model = tuner.get_best_models(num_models=1)[0]
loss, accuracy = best_model.evaluate(X_val, y_val)
print(f'Best Validation Accuracy: {accuracy:.2f}')

# Save the best model
best_model.save('cnn_1d_model_best.h5')
