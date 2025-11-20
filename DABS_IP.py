import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import json

# Parameters
IMG_SIZE = (227, 227)
BATCH_SIZE = 32
EPOCHS = 5
COW_HEALTH_DIR = r"C:\Users\HP\PycharmProjects\DABS\FilterDataset\cow\"  # Healthy / Diseased

def build_cow_disease_model():
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

def train_cow_disease_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        COW_HEALTH_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        COW_HEALTH_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    print("Class indices:", train_gen.class_indices)

    model = build_cow_disease_model()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("cow_disease_model.h5", save_best_only=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    model.save("final_cow_disease_model.h5")
    print("Model saved as final_cow_disease_model.h5")

    # Classification report
    val_gen.reset()
    val_images = []
    val_labels = []

    for _ in range(len(val_gen)):
        imgs, labels = next(val_gen)
        val_images.append(imgs)
        val_labels.append(labels)

    val_images = np.concatenate(val_images)
    val_labels = np.concatenate(val_labels)

    y_pred = (model.predict(val_images) > 0.5).astype("int32")

    report = classification_report(
        val_labels,
        y_pred,
        target_names=["Diseased", "Healthy"],
        output_dict=True
    )

    with open("cow_disease_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Classification report saved as cow_disease_report.json")
    return model

if __name__ == "__main__":
    print("Training Cow Disease Detection Model...")
    train_cow_disease_model()
