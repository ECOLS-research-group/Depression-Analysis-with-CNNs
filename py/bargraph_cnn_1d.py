# 1D_CNN.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import kerastuner as kt

# Constants
IMG_SIZE = 64
CATEGORIES = ['0', '1']
HEATMAPS_DIR = '/content/bar_graphs'

def load_images():
    images, labels = [], []
    
    for category in CATEGORIES:
        category_path = os.path.join(HEATMAPS_DIR, category)
        for filename in os.listdir(category_path):
            if filename.endswith('.png'):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img_array = img.flatten()
                images.append(img_array)
                labels.append(int(category))

    X = np.array(images) / 255.0
    y = np.array(labels)
    return X.reshape((X.shape[0], IMG_SIZE * IMG_SIZE, 1)), y

def build_1d_model(hp):
    model = Sequential()
    model.add(Conv1D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=16),
        kernel_size=hp.Int('kernel_size_1', min_value=3, max_value=7, step=2),
        activation='relu',
        input_shape=(IMG_SIZE * IMG_SIZE, 1)
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load images
    X, y = load_images()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune and train the 1D CNN model
    tuner = kt.BayesianOptimization(build_1d_model, objective='val_accuracy', max_trials=10)
    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    
    # Evaluate best model
    best_model_1d = tuner.get_best_models(num_models=1)[0]
    best_model_1d.evaluate(X_val, y_val)

if __name__ == "__main__":
    main()
