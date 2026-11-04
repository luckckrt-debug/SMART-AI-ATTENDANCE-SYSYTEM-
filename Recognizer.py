import os
import json
import warnings
from pathlib import Path
import cv2
import numpy as np
from keras.models import load_model

from FaceDetection.face_detection import face
from embedding import emb
from database_manager import db_manager

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "Face_recognition.MODEL"
MODEL_H5_PATH = MODEL_DIR / "Face_recognition.h5"
LABELS_PATH = MODEL_DIR / "labels.json"


class FaceRecognizer:
    """
    Handles live recognition and attendance marking for both web streaming and desktop OpenCV.
    """

    def __init__(self, confidence_threshold=0.85, consecutive_frames_needed=10):
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_needed = consecutive_frames_needed
        self.fd = face()
        self.e = emb()
        self.model = None
        self.labels_data = None
        self.session_counts = {}  # Tracks consecutive detection frames: {roll: frame_count}
        self.session_marked = set()  # Roll numbers already marked during current session
        self._load_recognition_model()

    def _load_recognition_model(self):
        if LABELS_PATH.exists():
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    self.labels_data = json.load(f)
            except Exception as e:
                print(f"[Recognizer] Error reading labels.json: {e}")

        model_file = None
        if MODEL_H5_PATH.exists():
            model_file = MODEL_H5_PATH
        elif MODEL_PATH.exists():
            model_file = MODEL_PATH

        if model_file:
            try:
                self.model = load_model(str(model_file), compile=False)
                print(f"[Recognizer] Loaded classifier model from {model_file.name}")
            except Exception as e:
                print(f"[Recognizer] Error loading model: {e}")
                self.model = None
        else:
            self.model = None

    def is_model_ready(self):
        return self.model is not None and self.labels_data is not None

    def reset_session(self):
        self.session_counts.clear()
        self.session_marked.clear()

    def process_frame(self, frame, subject="Hindi", auto_mark=True):
        """
        Processes a single BGR frame, detects faces, predicts identity,
        and optionally marks attendance in the database.
        Returns: (annotated_frame, list_of_detected_faces_info)
        """
        if frame is None or frame.size == 0:
            return frame, []

        subj = subject.strip().capitalize()
        if subj not in ["Hindi", "English"]:
            subj = "Hindi"

        annotated_frame = frame.copy()
        crops, coords = self.fd.detectFace(frame)
        detections = []

        if not self.is_model_ready():
            for (x, y, w, h) in coords:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    annotated_frame,
                    "Model Not Trained",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
            return annotated_frame, [{"status": "Model Not Trained"}]

        idx_to_label = self.labels_data.get("idx_to_label", {})
        idx_to_roll = self.labels_data.get("idx_to_roll", {})
        idx_to_name = self.labels_data.get("idx_to_name", {})

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue

            x, y, w, h = coords[i]
            prep = self.e.preprocess_face(crop)
            if prep is None:
                continue

            emb_feat = self.e.calculate(prep)
            emb_input = np.expand_dims(emb_feat, axis=0)
            predictions = self.model.predict(emb_input, verbose=0)[0]

            best_idx = int(np.argmax(predictions))
            confidence = float(predictions[best_idx])
            str_idx = str(best_idx)

            if confidence >= self.confidence_threshold and str_idx in idx_to_roll:
                roll = idx_to_roll[str_idx]
                name = idx_to_name[str_idx]
                display_label = f"{name} ({int(confidence * 100)}%)"

                # Sustained detection count for auto-marking attendance
                cur_count = self.session_counts.get(roll, 0) + 1
                self.session_counts[roll] = cur_count

                already_marked = roll in self.session_marked
                marked_now = False

                if auto_mark and cur_count >= self.consecutive_frames_needed and not already_marked:
                    success, msg, _ = db_manager.mark_attendance(subj, roll, name)
                    self.session_marked.add(roll)
                    marked_now = True

                color = (0, 255, 0) if (already_marked or marked_now) else (255, 200, 0)
                status_text = "Marked" if (already_marked or marked_now) else f"Hold... ({cur_count}/{self.consecutive_frames_needed})"

                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    annotated_frame,
                    f"{display_label} - {status_text}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                detections.append({
                    "roll": roll,
                    "name": name,
                    "confidence": round(confidence, 3),
                    "marked": already_marked or marked_now,
                    "status": status_text,
                    "subject": subj
                })
            else:
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    annotated_frame,
                    f"Unknown ({int(confidence * 100)}%)",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
                detections.append({
                    "roll": "Unknown",
                    "name": "Unknown Student",
                    "confidence": round(confidence, 3),
                    "marked": False,
                    "status": "Unknown",
                    "subject": subj
                })

        return annotated_frame, detections


def Recognition(subject="Hindi", camera_index=0):
    """
    Standalone desktop camera recognition loop for backward compatibility.
    """
    subj = "Hindi" if str(subject).lower() in ["hindi", "1"] else "English"
    rec = FaceRecognizer()

    if not rec.is_model_ready():
        print("[Recognizer] Error: Model is not ready. Train the model first.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[Recognizer] Error: Cannot access camera index {camera_index}.")
        return

    print(f"[Recognizer] Starting {subj} attendance. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            annotated, _ = rec.process_frame(frame, subject=subj)

            cv2.imshow(f"{subj} Attendance - Press 'q' to Finish", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db_manager.export_csv(subj)
        print(f"[Recognizer] Finished {subj} attendance session.")
