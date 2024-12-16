# Purpose : Train an emotion classification model using FER-2013 and AffectNet datasets.


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_data_dir = "notebook\data\AffectNet\train"
val_data_dir = "notebook\data\AffectNet\val"
model_path = "./models/emotion_classifier/emotion_model.h5"

# Data generator
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(train_data_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64)
val_gen = datagen.flow_from_directory(val_data_dir, target_size=(48, 48), color_mode='grayscale', batch_size=64)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # Emotion categories
])

# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=50)

# Save model
model.save(model_path)