import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Parameters
IMG_SIZE = (227, 227)
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "C:\\Users\\HP\\PycharmProjects\\DABS\\Lumpy\\Lumpy"  # Should have folders: cow, not_cow

# Build the Cow vs Not-Cow Model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# Train and Save the Model
def train_and_save():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = build_model()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("best_prefilter_model.h5", save_best_only=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    model.save("prefilter_cow_model.h5")
    print("Model saved as prefilter_cow_model.h5")

if __name__ == "__main__":
    train_and_save()
