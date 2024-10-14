import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

# Parameters
base_dir = 'path/to/AlgX3_64x64_merged_tensors_with_labels'  # Change this path
img_size = 64
batch_size = 32
epochs = 50

# Data Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Split the data into 80% training and 20% validation
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Define model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Conv2D(hp.Int('filters1', min_value=32, max_value=64, step=32), (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(hp.Int('filters2', min_value=64, max_value=128, step=64), (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(hp.Int('units', min_value=256, max_value=512, step=128), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize Keras Tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='keras_tuner',
    project_name='cnn_tuning'
)

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Search for the best hyperparameters
tuner.search(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stopping]
)

# Get and evaluate the best model
best_model = tuner.get_best_models(num_models=1)[0]
val_loss, val_accuracy = best_model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Save the best model
best_model.save('cnn_2d_model_best.h5')
