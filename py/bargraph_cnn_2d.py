# 2D_CNN.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt

# Constants
IMG_SIZE = 64
BATCH_SIZE = 32
CATEGORIES = ['0', '1']
HEATMAPS_DIR = '/content/bar_graphs'

def build_2d_model(hp):
    model = Sequential([
        Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(hp.Int('units', min_value=256, max_value=512, step=128), activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Data Generator
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        HEATMAPS_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        HEATMAPS_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # Tune and train the 2D CNN model
    tuner_2d = kt.BayesianOptimization(build_2d_model, objective='val_accuracy', max_trials=10)
    tuner_2d.search(train_generator, validation_data=validation_generator, epochs=50)

    # Evaluate best 2D model
    best_model_2d = tuner_2d.get_best_models(num_models=1)[0]
    best_model_2d.evaluate(validation_generator)

if __name__ == "__main__":
    main()
