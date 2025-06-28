# train_mask_detector.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
import pickle

# Initialize parameters
INIT_LR = 1e-4
EPOCHS = 50
BS = 32

# Directory and categories
DIRECTORY = "dataset"
CATEGORIES = ["with_mask", "without_mask"]


def load_dataset():
    print("[INFO] loading images...")
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)
                data.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    # Convert and encode labels
    lb = LabelBinarizer()
    labels = to_categorical(lb.fit_transform(labels))
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels, lb


def build_model():
    baseModel = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze base layers
    for layer in baseModel.layers:
        layer.trainable = False

    return model


def train_model():
    # Load data
    data, labels, lb = load_dataset()

    # Split data
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"[INFO] Training samples: {len(trainX)}, Validation samples: {len(testX)}")

    # Data augmentation
    train_aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Build model
    model = build_model()

    # Phase 1: Train head only
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True)
    ]

    print("[INFO] training head...")
    H1 = model.fit(
        train_aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS // 2,
        callbacks=callbacks
    )

    # Phase 2: Fine-tune
    print("[INFO] preparing for fine-tuning...")
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR / 10),
        metrics=["accuracy"]
    )

    H2 = model.fit(
        train_aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        initial_epoch=H1.epoch[-1] + 1,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save history for plotting
    history = {
        'loss': H1.history['loss'] + H2.history['loss'],
        'val_loss': H1.history['val_loss'] + H2.history['val_loss'],
        'accuracy': H1.history['accuracy'] + H2.history['accuracy'],
        'val_accuracy': H1.history['val_accuracy'] + H2.history['val_accuracy']
    }

    with open("training_history.pkl", "wb") as f:
        pickle.dump(history, f)

    # Evaluate and save final model
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

    print("[INFO] saving mask detector model...")
    model.save("mask_detector_final.keras")

    return history


if __name__ == "__main__":
    train_model()
