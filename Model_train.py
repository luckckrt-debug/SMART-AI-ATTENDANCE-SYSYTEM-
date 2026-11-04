import os
import re
import json
import warnings
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.utils import to_categorical

from Model_architecture.modelArch import DenseArchs
from embedding import emb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_DIR = BASE_DIR / "people"
MODEL_DIR = BASE_DIR / "Model"
LABELS_PATH = MODEL_DIR / "labels.json"
MODEL_PATH = MODEL_DIR / "Face_recognition.MODEL"

# Global training state for web UI polling
TRAINING_STATE = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 60,
    "status": "Idle",
    "loss": 0.0,
    "accuracy": 0.0,
    "error": None,
    "classes": []
}


def parse_student_folder(folder_name):
    """
    Parses folder name into (roll_number, student_name).
    Supports formats:
      - '21_BHAVESH' or '21-BHAVESH'
      - '21BHAVESH' or '1bhavesh'
      - 'kushagra' (fallback)
    """
    match = re.match(r"^(\d+)[_\s-]*(.*)$", folder_name)
    if match:
        roll = match.group(1)
        name = match.group(2) if match.group(2) else f"Student_{roll}"
        return roll, name
    return folder_name, folder_name


def Model_Training(epochs=60, batch_size=16, learning_rate=0.001, progress_callback=None):
    """
    Trains the classifier on face embeddings and saves the model + label mapping.
    """
    global TRAINING_STATE

    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["status"] = "Preparing dataset..."
    TRAINING_STATE["progress"] = 5
    TRAINING_STATE["error"] = None
    TRAINING_STATE["total_epochs"] = epochs

    try:
        if not PEOPLE_DIR.exists():
            raise FileNotFoundError("People directory does not exist. Please enroll students first.")

        folders = sorted([f for f in os.listdir(PEOPLE_DIR) if (PEOPLE_DIR / f).is_dir()])
        # Filter folders that have at least 1 image
        valid_folders = []
        for f in folders:
            images = [img for img in os.listdir(PEOPLE_DIR / f) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) > 0:
                valid_folders.append(f)

        if len(valid_folders) == 0:
            raise ValueError("No student image data found in people/ directory.")

        n_classes = len(valid_folders)
        actual_classes = max(2, n_classes)

        # Build label mappings
        class_to_idx = {folder: idx for idx, folder in enumerate(valid_folders)}
        idx_to_label = {}
        idx_to_roll = {}
        idx_to_name = {}

        for folder, idx in class_to_idx.items():
            roll, name = parse_student_folder(folder)
            idx_to_label[str(idx)] = f"{name} ({roll})"
            idx_to_roll[str(idx)] = roll
            idx_to_name[str(idx)] = name

        labels_data = {
            "classes": valid_folders,
            "class_to_idx": class_to_idx,
            "idx_to_label": idx_to_label,
            "idx_to_roll": idx_to_roll,
            "idx_to_name": idx_to_name,
            "n_classes": n_classes
        }

        TRAINING_STATE["classes"] = list(idx_to_label.values())
        TRAINING_STATE["status"] = "Extracting FaceNet embeddings..."
        TRAINING_STATE["progress"] = 15

        e = emb()
        x_data = []
        y_data = []

        total_images = sum(len(os.listdir(PEOPLE_DIR / f)) for f in valid_folders)
        processed = 0

        for folder in valid_folders:
            folder_path = PEOPLE_DIR / folder
            class_idx = class_to_idx[folder]
            img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            for img_name in img_files:
                img_path = folder_path / img_name
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                prep = e.preprocess_face(img)
                if prep is None:
                    continue

                embs = e.calculate(prep)
                x_data.append(embs)
                y_data.append(class_idx)

                processed += 1
                if total_images > 0:
                    prog = 15 + int((processed / total_images) * 40)
                    TRAINING_STATE["progress"] = min(prog, 55)

        if len(x_data) == 0:
            raise ValueError("Could not extract any valid face embeddings from student images.")

        X = np.array(x_data, dtype='float32')
        y_arr = np.array(y_data, dtype='int32')

        # Handle edge case where only 1 student is enrolled
        if n_classes == 1:
            # Duplicate with dummy class to allow softmax training
            dummy_emb = np.random.normal(0, 0.05, (10, 128)).astype('float32')
            X = np.vstack([X, dummy_emb])
            y_arr = np.concatenate([y_arr, np.full((10,), 1, dtype='int32')])
            idx_to_label["1"] = "Unknown"
            labels_data["idx_to_label"] = idx_to_label

        y = to_categorical(y_arr, num_classes=actual_classes)

        TRAINING_STATE["status"] = "Compiling neural architecture..."
        TRAINING_STATE["progress"] = 60

        arc = DenseArchs(actual_classes)
        face_model = arc.arch()
        optimizer = Adam(learning_rate=learning_rate)
        face_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        TRAINING_STATE["status"] = "Training model..."

        class ProgressCallback:
            def __call__(self, epoch, logs):
                TRAINING_STATE["current_epoch"] = epoch + 1
                TRAINING_STATE["loss"] = round(float(logs.get("loss", 0)), 4)
                TRAINING_STATE["accuracy"] = round(float(logs.get("accuracy", 0)), 4)
                ep_prog = 60 + int(((epoch + 1) / epochs) * 35)
                TRAINING_STATE["progress"] = min(ep_prog, 95)
                if progress_callback:
                    progress_callback(TRAINING_STATE)

        from keras.callbacks import LambdaCallback
        cb = LambdaCallback(on_epoch_end=lambda ep, logs: ProgressCallback()(ep, logs))

        face_model.fit(
            X, y,
            batch_size=min(batch_size, len(X)),
            epochs=epochs,
            shuffle=True,
            verbose=0,
            callbacks=[cb]
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # Ensure older file is safely replaced as HDF5 format
        if MODEL_PATH.exists() and not MODEL_PATH.is_dir():
            MODEL_PATH.unlink()
        face_model.save(str(MODEL_PATH), save_format='h5')
        # Also save as standard .h5
        face_model.save(str(MODEL_DIR / "Face_recognition.h5"), save_format='h5')
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels_data, f, indent=4)

        TRAINING_STATE["progress"] = 100
        TRAINING_STATE["status"] = f"Training completed successfully ({len(X)} samples, {n_classes} classes)."
        TRAINING_STATE["is_training"] = False
        print(f"[ModelTrain] {TRAINING_STATE['status']}")
        return True, TRAINING_STATE["status"]

    except Exception as e:
        TRAINING_STATE["is_training"] = False
        TRAINING_STATE["status"] = f"Training failed: {str(e)}"
        TRAINING_STATE["error"] = str(e)
        print(f"[ModelTrain] Error: {e}")
        return False, str(e)
