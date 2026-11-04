import time
import json
import base64
import datetime
import threading
from pathlib import Path
import cv2
import numpy as np

from ai_engine import face_detector, align_face_crop, assess_face_quality, prototype_matcher, faculty_matcher
from embedding import emb
from database_manager import db_manager

BASE_DIR = Path(__file__).resolve().parent
LABELS_PATH = BASE_DIR / "Model" / "labels.json"
MODEL_H5_PATH = BASE_DIR / "Model" / "Face_recognition.h5"
MODEL_PATH = BASE_DIR / "Model" / "Face_recognition.MODEL"


class CentroidTracker:
    """Lightweight 2D centroid tracker for multi-face association across frames."""
    def __init__(self, max_disappeared=15):
        self.next_object_id = 1
        self.objects = {}  # id: (cx, cy)
        self.disappeared = {}  # id: count
        self.bboxes = {}  # id: [x, y, w, h]
        self.max_disappeared = max_disappeared

    def register(self, centroid, bbox):
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.bboxes[obj_id] = bbox
        self.disappeared[obj_id] = 0
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id):
        self.objects.pop(obj_id, None)
        self.bboxes.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def update(self, rects):
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.bboxes

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i])
            return self.bboxes

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for (row, col) in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > 180:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.bboxes[obj_id] = rects[col]
            self.disappeared[obj_id] = 0
            assigned_rows.add(row)
            assigned_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(assigned_rows)
        unused_cols = set(range(0, D.shape[1])).difference(assigned_cols)

        for row in unused_rows:
            obj_id = object_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self.deregister(obj_id)

        for col in unused_cols:
            self.register(input_centroids[col], rects[col])

        return self.bboxes


class ClassroomCamera:
    """
    Classroom camera processing engine.
    Supports simultaneous multi-person detection, biometric consent filtering,
    continuous presence tracking, camera health monitoring, and reconnects.
    """

    def __init__(self, camera_index=0, confidence_threshold=0.85, consecutive_frames_needed=6):
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_needed = consecutive_frames_needed

        self.e = emb()
        self.model = None
        self.labels_data = None

        # Camera health metrics
        self.status = "OFFLINE"
        self.fps = 0.0
        self.latency_ms = 0.0
        self.faces_in_view = 0
        self.total_frames = 0
        self.last_frame_time = 0.0
        self.reconnect_attempts = 0

        # Frame counters for sustained presence per student
        self.detection_streak = {}
        self.active_session_id = None

        # Multi-person tracker instance
        self.tracker = CentroidTracker(max_disappeared=15)

        # Internal capture thread & lock
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.worker_thread = None

        self.load_models()

    def load_models(self):
        """Loads classifier weights, student prototypes, and faculty prototypes."""
        prototype_matcher.load_prototypes()
        faculty_matcher.load_prototypes()

        if LABELS_PATH.exists():
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    self.labels_data = json.load(f)
            except Exception as e:
                print(f"[ClassroomCamera] Error loading labels.json: {e}")

        model_file = None
        if MODEL_H5_PATH.exists():
            model_file = MODEL_H5_PATH
        elif MODEL_PATH.exists():
            model_file = MODEL_PATH

        if model_file:
            try:
                from keras.models import load_model
                self.model = load_model(str(model_file), compile=False)
                print(f"[ClassroomCamera] Loaded classifier model: {model_file.name}")
            except Exception as e:
                print(f"[ClassroomCamera] Error loading classifier: {e}")
                self.model = None

    def is_model_ready(self):
        return (len(prototype_matcher.prototypes) > 0) or (len(faculty_matcher.prototypes) > 0) or (self.model is not None and self.labels_data is not None)

    def get_health(self):
        """Returns real-time camera health and processing metrics."""
        curr_status = self.status
        if time.time() - self.last_frame_time > 4.0:
            curr_status = "OFFLINE" if not self.is_running else "STANDBY"
        return {
            "status": curr_status,
            "fps": round(self.fps, 1),
            "latency_ms": round(self.latency_ms, 1),
            "faces_in_view": self.faces_in_view,
            "total_frames": self.total_frames,
            "camera_index": self.camera_index,
            "model_ready": self.is_model_ready(),
            "model_version": prototype_matcher.active_version,
            "confidence_threshold": self.confidence_threshold,
            "frames_needed": self.consecutive_frames_needed
        }

    def start_camera(self, camera_index=None):
        """Starts background hardware camera capture thread."""
        if camera_index is not None:
            self.camera_index = camera_index

        if self.is_running:
            return True

        self.is_running = True
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()
        return True

    def stop_camera(self):
        """Stops hardware camera thread and releases device."""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.status = "OFFLINE"

    def _capture_loop(self):
        """Continuous hardware camera capture loop with auto-reconnect."""
        self.status = "CONNECTING"
        self.cap = cv2.VideoCapture(self.camera_index)

        consecutive_failures = 0
        while self.is_running:
            start_t = time.time()
            if not self.cap.isOpened() or consecutive_failures > 5:
                self.status = "RECONNECTING"
                self.reconnect_attempts += 1
                time.sleep(1.5)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    consecutive_failures = 0
                    self.status = "ONLINE"
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                time.sleep(0.1)
                continue

            consecutive_failures = 0
            self.status = "ONLINE"
            frame = cv2.flip(frame, 1)

            # Process multi-face recognition
            annotated, detections = self.process_frame(frame, auto_mark=True)

            dur = time.time() - start_t
            self.latency_ms = dur * 1000.0
            self.fps = 1.0 / max(0.001, dur)
            self.total_frames += 1

            with self.lock:
                self.current_frame = annotated

            time.sleep(0.02)

    def get_latest_frame_jpeg(self):
        """Returns JPEG encoded bytes of latest hardware camera frame."""
        with self.lock:
            if self.current_frame is None:
                return None
            ret, buffer = cv2.imencode(".jpg", self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes() if ret else None

    # =========================================================================
    # MULTI-PERSON PROCESSING (Client-side & Server-side shared)
    # =========================================================================
    def process_frame(self, frame, auto_mark=True, active_session_id=None, session_id=None):
        """
        Simultaneously detects multiple faces in the frame.
        1. Multi-face detection and tracking.
        2. Quality assessment (>= 45x45, blur check).
        3. Priority Faculty Exclusion: Check faculty prototypes first; if matched, mark as VERIFIED_FACULTY, blue box, exclude from student attendance.
        4. Priority Class-Wise Eligibility: Check student against snapshotted eligible_student_ids for active session.
           - If matched with high confidence and sustained streak -> mark attendance (VERIFIED_STUDENT / ATTENDANCE_MARKED), green box.
        5. Outside-Class Person Detection: If not eligible in active class session, check against institution student prototypes.
           - If matched -> OUTSIDE_CLASS warning, orange box, DO NOT mark attendance.
        6. Unknown / Ambiguous Handling:
           - If ambiguous or low quality -> yellow box.
           - If unregistered / unknown -> red box, warning overlay, DO NOT mark attendance.
        7. Full audit trail logged to Recognition_Logs table.
        """
        if frame is None or frame.size == 0:
            return frame, []

        active_session_id = session_id or active_session_id
        start_time = time.time()
        self.status = "ONLINE"
        self.last_frame_time = time.time()
        self.total_frames += 1

        annotated = frame.copy()

        # 1. Advanced Multi-Face Detection
        detected_faces = face_detector.detect(frame)
        coords = [f["bbox"] for f in detected_faces]
        self.faces_in_view = len(detected_faces)

        # 2. Track bounding boxes across frames
        tracked_boxes = self.tracker.update(coords)
        track_map = {}
        for tid, box in tracked_boxes.items():
            for idx, c in enumerate(coords):
                if c[0] == box[0] and c[1] == box[1] and c[2] == box[2] and c[3] == box[3]:
                    track_map[idx] = f"Track #{tid}"
                    break

        detections = []
        sess = active_session_id or self.active_session_id
        if not sess:
            active_rec = db_manager.get_active_session()
            sess = active_rec["session_id"] if active_rec else None
            self.active_session_id = sess

        # Query snapshotted eligible students for active session
        eligible_ids = db_manager.get_eligible_students_for_session(sess) if sess else None

        # If model is not ready, display informative alert box
        if not self.is_model_ready():
            for (x, y, w, h) in coords:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(
                    annotated, "Model Not Trained",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2
                )
            self._draw_hud(annotated, "OFFLINE - MODEL NOT TRAINED")
            return annotated, []

        for i, f_info in enumerate(detected_faces):
            x, y, w, h = f_info["bbox"]
            landmarks = f_info.get("landmarks")

            # 3. 5-point alignment & crop
            aligned = align_face_crop(frame, [x, y, w, h], landmarks, output_size=(160, 160))
            if aligned is None or aligned.size == 0:
                continue

            # 4. Face quality check
            quality = assess_face_quality(aligned, [x, y, w, h])
            q_score = quality.get("quality_score", 1.0)
            is_acceptable = quality.get("acceptable", True)
            if not is_acceptable or min(w, h) < 40:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (8, 179, 234), 2)
                cv2.putText(
                    annotated, "Low Quality Face",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (8, 179, 234), 2
                )
                db_manager.log_recognition_decision(
                    session_id=sess, person_id="UNKNOWN", person_type="STUDENT",
                    decision_state="LOW_QUALITY", confidence=0.0,
                    metadata={"reason": "Face image blurry or too small"}
                )
                detections.append({
                    "roll": "UNKNOWN",
                    "name": "Low Quality Face",
                    "confidence": 0.0,
                    "status": "Low Quality",
                    "recognition_state": "LOW_QUALITY",
                    "is_faculty": False,
                    "is_outside_class": False,
                    "color": "#eab308",
                    "warning_message": None,
                    "track_id": track_map.get(i, f"Track #{i + 1}"),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": [int(x + w / 2), int(y + h / 2)]
                })
                continue

            # 5. Extract FaceNet embedding
            prep = self.e.preprocess_face(aligned)
            if prep is None:
                continue

            emb_feat = self.e.calculate(prep)
            emb_norm = max(1e-9, float(np.linalg.norm(emb_feat)))
            emb_feat = emb_feat / emb_norm

            # =================================================================
            # STEP A: FACULTY EXCLUSION SYSTEM (Faculty is checked first)
            # =================================================================
            fac_id, fac_name, fac_conf, fac_state = faculty_matcher.match_embedding(emb_feat)
            if fac_state == "VERIFIED_FACULTY":
                # Faculty member identified -> Excluded from student attendance
                # Pure Blue: BGR (246, 130, 59) -> #3b82f6
                box_color = (246, 130, 59)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                cv2.rectangle(annotated, (x, max(0, y - 26)), (x + w, y), box_color, -1)
                cv2.putText(
                    annotated, f"FACULTY: {fac_name}",
                    (x + 4, max(14, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2
                )
                cv2.putText(
                    annotated, f"Attendance Excluded ({int(fac_conf * 100)}%)",
                    (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1
                )

                db_manager.log_recognition_decision(
                    session_id=sess, person_id=fac_id, person_type="FACULTY",
                    decision_state="VERIFIED_FACULTY", confidence=fac_conf,
                    metadata={"name": fac_name, "excluded": True}
                )

                detections.append({
                    "roll": fac_id,
                    "name": fac_name,
                    "confidence": round(fac_conf, 3),
                    "status": "Faculty Excluded",
                    "recognition_state": "VERIFIED_FACULTY",
                    "is_faculty": True,
                    "is_outside_class": False,
                    "color": "#3b82f6",
                    "box_color": [59, 130, 246],
                    "warning_message": None,
                    "track_id": track_map.get(i, f"Track #{i + 1}"),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": [int(x + w / 2), int(y + h / 2)]
                })
                continue

            # =================================================================
            # STEP B: CLASS-WISE STUDENT RECOGNITION (Restricted to Eligible Roster)
            # =================================================================
            matched_roll, student_name, confidence, recog_state = prototype_matcher.match_embedding(
                emb_feat, eligible_ids=eligible_ids
            )

            # Fallback to legacy MLP model only if prototypes empty
            if not prototype_matcher.prototypes and self.model is not None and self.labels_data is not None:
                emb_input = np.expand_dims(emb_feat, axis=0)
                predictions = self.model.predict(emb_input, verbose=0)[0]
                best_idx = int(np.argmax(predictions))
                confidence = float(predictions[best_idx])
                str_idx = str(best_idx)
                idx_to_roll = self.labels_data.get("idx_to_roll", {})
                idx_to_name = self.labels_data.get("idx_to_name", {})
                if confidence >= self.confidence_threshold and str_idx in idx_to_roll:
                    matched_roll = idx_to_roll[str_idx]
                    student_name = idx_to_name.get(str_idx, f"Student {matched_roll}")
                    recog_state = "HIGH_CONFIDENCE"
                elif confidence >= 0.60:
                    matched_roll = idx_to_roll.get(str_idx, "UNKNOWN")
                    student_name = idx_to_name.get(str_idx, "Uncertain")
                    recog_state = "REVIEW_REQUIRED"
                else:
                    matched_roll = "UNKNOWN"
                    student_name = "Visitor / Unregistered"
                    recog_state = "UNKNOWN"

            # Case 1: Verified Eligible Student
            if recog_state in ["HIGH_CONFIDENCE", "REVIEW_REQUIRED"] and matched_roll != "UNKNOWN":
                student = db_manager.get_student(matched_roll)

                # Privacy Safeguard: Non-consenting
                if not student or not student.get("consent_given", 1):
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (140, 140, 140), 2)
                    cv2.putText(
                        annotated, "Identity Protected (Non-Consenting)",
                        (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
                    )
                    detections.append({
                        "roll": "ANONYMOUS",
                        "name": "Identity Protected",
                        "confidence": round(confidence, 3),
                        "status": "Consent Restricted",
                        "recognition_state": "ANONYMOUS",
                        "is_faculty": False,
                        "is_outside_class": False,
                        "color": "#94a3b8",
                        "is_consenting": False,
                        "track_id": track_map.get(i, f"Track #{i + 1}"),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "centroid": [int(x + w / 2), int(y + h / 2)]
                    })
                    continue

                student_name = student.get("name", student_name)
                stream_dept = student.get("stream") or student.get("department", "")

                streak = self.detection_streak.get(matched_roll, 0) + 1
                self.detection_streak[matched_roll] = streak

                event_action = "ACTIVE"
                is_verified = (streak >= self.consecutive_frames_needed and recog_state == "HIGH_CONFIDENCE")
                decision_state = "VERIFIED_STUDENT"

                if auto_mark and sess and is_verified:
                    success, msg, event_action = db_manager.record_session_presence(
                        sess, matched_roll, verification_method="AI_FACENET_CALIBRATED"
                    )
                    decision_state = "ATTENDANCE_MARKED" if event_action == "CHECKIN" else (
                        "DUPLICATE_ATTEMPT" if "Already" in msg else "VERIFIED_STUDENT"
                    )

                # Emerald Green: BGR (19, 185, 16) -> #10b981. Verifying Yellow: BGR (8, 179, 234)
                box_color = (19, 185, 16) if is_verified else (8, 179, 234)
                status_label = "Verified Present" if is_verified else (
                    f"Verifying ({streak}/{self.consecutive_frames_needed})" if recog_state == "HIGH_CONFIDENCE" else "Review Required"
                )

                cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                cv2.rectangle(annotated, (x, max(0, y - 24)), (x + w, y), box_color, -1)
                cv2.putText(
                    annotated, f"{student_name} ({matched_roll}) - {int(confidence * 100)}%",
                    (x + 4, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 25, 47), 2
                )
                cv2.putText(
                    annotated, f"{stream_dept} | {status_label}",
                    (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1
                )

                db_manager.log_recognition_decision(
                    session_id=sess, person_id=matched_roll, person_type="STUDENT",
                    decision_state=decision_state, confidence=confidence,
                    metadata={"name": student_name, "verified": is_verified, "streak": streak}
                )

                detections.append({
                    "roll": matched_roll,
                    "name": student_name,
                    "stream": stream_dept,
                    "department": stream_dept,
                    "profile_photo": student.get("profile_photo") or f"/static/profiles/{matched_roll}.jpg",
                    "confidence": round(confidence, 3),
                    "status": status_label,
                    "recognition_state": decision_state,
                    "event_action": event_action,
                    "verified": is_verified,
                    "is_faculty": False,
                    "is_outside_class": False,
                    "color": "#10b981" if is_verified else "#eab308",
                    "box_color": [16, 185, 129] if is_verified else [234, 179, 8],
                    "warning_message": None,
                    "is_consenting": True,
                    "quality_score": q_score,
                    "track_id": track_map.get(i, f"Track #{i + 1}"),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": [int(x + w / 2), int(y + h / 2)]
                })

            else:
                # =============================================================
                # STEP C: CHECK IF STUDENT BELONGS TO ANOTHER CLASS (OUTSIDE-CLASS)
                # =============================================================
                out_roll, out_name, out_conf, out_state = prototype_matcher.match_embedding(
                    emb_feat, eligible_ids=None
                )

                if out_state in ["HIGH_CONFIDENCE", "REVIEW_REQUIRED"] and out_roll != "UNKNOWN":
                    # This person is a known student in the institution, but not eligible for current session!
                    # Orange Box: BGR (22, 115, 249) -> #f97316
                    box_color = (22, 115, 249)
                    warn_msg = "⚠ You are not part of this class. Please leave the classroom or contact the faculty member."

                    cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                    cv2.rectangle(annotated, (x, max(0, y - 24)), (x + w, y), box_color, -1)
                    cv2.putText(
                        annotated, f"NOT IN THIS CLASS ({out_roll})",
                        (x + 4, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2
                    )
                    cv2.putText(
                        annotated, f"{out_name} - Attendance Excluded",
                        (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1
                    )

                    db_manager.log_recognition_decision(
                        session_id=sess, person_id=out_roll, person_type="STUDENT",
                        decision_state="OUTSIDE_CLASS", confidence=out_conf,
                        metadata={"name": out_name, "class_required": sess, "action": "Attendance Denied"}
                    )

                    detections.append({
                        "roll": out_roll,
                        "name": out_name,
                        "confidence": round(out_conf, 3),
                        "status": "Outside Class",
                        "recognition_state": "OUTSIDE_CLASS",
                        "is_faculty": False,
                        "is_outside_class": True,
                        "color": "#f97316",
                        "box_color": [249, 115, 22],
                        "warning_message": warn_msg,
                        "quality_score": q_score,
                        "track_id": track_map.get(i, f"Track #{i + 1}"),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "centroid": [int(x + w / 2), int(y + h / 2)]
                    })

                elif out_state == "AMBIGUOUS" or recog_state == "AMBIGUOUS":
                    # Ambiguous match (margin between top-2 candidates < 0.04) -> Yellow Box
                    box_color = (8, 179, 234)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(
                        annotated, "AMBIGUOUS MATCH",
                        (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2
                    )
                    db_manager.log_recognition_decision(
                        session_id=sess, person_id="AMBIGUOUS", person_type="STUDENT",
                        decision_state="AMBIGUOUS", confidence=round(out_conf, 3),
                        metadata={"reason": "Similarity difference between candidates below margin threshold"}
                    )
                    detections.append({
                        "roll": "AMBIGUOUS",
                        "name": "Ambiguous Identity",
                        "confidence": round(out_conf, 3),
                        "status": "Ambiguous - Attendance Denied",
                        "recognition_state": "AMBIGUOUS",
                        "is_faculty": False,
                        "is_outside_class": False,
                        "color": "#eab308",
                        "box_color": [234, 179, 8],
                        "warning_message": None,
                        "quality_score": q_score,
                        "track_id": track_map.get(i, f"Track #{i + 1}"),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "centroid": [int(x + w / 2), int(y + h / 2)]
                    })

                else:
                    # =========================================================
                    # STEP D: UNREGISTERED / UNKNOWN PERSON
                    # =========================================================
                    # Red Box: BGR (68, 68, 239) -> #ef4444
                    box_color = (68, 68, 239)
                    warn_msg = "Unknown person detected — Attendance not marked."

                    cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
                    cv2.putText(
                        annotated, f"UNKNOWN PERSON ({int(out_conf * 100)}%)",
                        (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2
                    )

                    db_manager.log_recognition_decision(
                        session_id=sess, person_id="UNKNOWN", person_type="UNKNOWN",
                        decision_state="UNKNOWN", confidence=round(out_conf, 3),
                        metadata={"action": "Attendance Denied"}
                    )

                    detections.append({
                        "roll": "UNKNOWN",
                        "name": "UNKNOWN PERSON",
                        "confidence": round(out_conf, 3),
                        "status": "Unregistered",
                        "recognition_state": "UNKNOWN",
                        "is_faculty": False,
                        "is_outside_class": False,
                        "is_unknown": True,
                        "color": "#ef4444",
                        "box_color": [239, 68, 68],
                        "warning_message": warn_msg,
                        "is_consenting": False,
                        "quality_score": q_score,
                        "track_id": track_map.get(i, f"Track #{i + 1}"),
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "centroid": [int(x + w / 2), int(y + h / 2)]
                    })

        self._draw_hud(annotated, f"CLASSROOM CAM: ONLINE | FACES: {len(detected_faces)}")
        self.latency_ms = (time.time() - start_time) * 1000.0
        return annotated, detections

    def _draw_hud(self, img, title):
        """Draws cyan HUD overlay on the frame."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (10, 25, 47), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        color = (0, 255, 128) if "ONLINE" in title or "LIVE" in title else (0, 165, 255)
        cv2.circle(img, (16, 16), 5, color, -1)
        cv2.putText(img, title, (28, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 241, 255), 1)

        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(img, now_str, (w - 75, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 242, 254), 1)


# Global classroom camera instance
classroom_camera = ClassroomCamera()
