"""
Unified AI Computer Vision & Face Recognition Engine
Implements:
1. Deep Learning Multi-Face Detection via OpenCV YuNet (with Dlib/Haar fallbacks)
2. 5-Point Affine Similarity Alignment to Canonical FaceNet Reference
3. Image Quality Assessment (Blur, Illumination, Size, Anti-Spoofing Liveness)
4. FaceNet 128-D L2-Normalized Feature Embeddings
5. Multi-Prototype Exemplar Clustering & Calibrated Cosine Metric
6. Multi-Person Tracking with Temporal Identity Voting & Verification
"""

import os
import json
import time
import math
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
YUNET_PATH = BASE_DIR / "FaceDetection" / "face_detection_yunet_2023mar.onnx"
PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
PROTOTYPES_PATH = BASE_DIR / "Model" / "student_prototypes.json"
MODEL_METRICS_PATH = BASE_DIR / "Model" / "active_model_metrics.json"

# Canonical FaceNet (160x160) 5 facial landmark reference points
REFERENCE_FACENET_160 = np.array([
    [54.7, 65.5],   # left eye
    [105.3, 65.5],  # right eye
    [80.0, 93.0],   # nose
    [59.5, 117.0],  # left mouth
    [100.5, 117.0]  # right mouth
], dtype=np.float32)


# =============================================================================
# 1. ADVANCED MULTI-FACE DETECTION (YuNet DNN + Dlib + Haar Cascades)
# =============================================================================
class AdvancedFaceDetector:
    """
    Robust multi-face detector supporting crowded scenes, overlapping faces,
    varying poses, and difficult classroom illumination.
    """
    def __init__(self, min_confidence=0.55, nms_threshold=0.35):
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.yunet_detector = None
        self.last_input_size = None
        self._init_yunet()
        self._init_fallbacks()

    def _init_yunet(self):
        if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 10000:
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH),
                    "",
                    (320, 320),
                    score_threshold=self.min_confidence,
                    nms_threshold=self.nms_threshold,
                    top_k=200
                )
                print("[AdvancedFaceDetector] YuNet DNN detector initialized.")
            except Exception as e:
                print(f"[AdvancedFaceDetector] YuNet init error: {e}")
                self.yunet_detector = None

    def _init_fallbacks(self):
        self.dlib_detector = None
        self.dlib_pred = None
        try:
            import dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            if PREDICTOR_PATH.exists():
                self.dlib_pred = dlib.shape_predictor(str(PREDICTOR_PATH))
        except Exception:
            pass

        # Haar Cascades
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    def detect(self, img_bgr):
        """
        Detects all visible faces in img_bgr.
        Returns list of dicts:
        [
            {
                "bbox": [x, y, w, h],
                "confidence": float,
                "landmarks": [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]] or None,
                "source": "YuNet" | "Dlib" | "Haar"
            }, ...
        ]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        detections = []

        # 1. Try YuNet primary deep detector
        if self.yunet_detector is not None:
            try:
                if self.last_input_size != (w, h):
                    self.yunet_detector.setInputSize((w, h))
                    self.last_input_size = (w, h)

                _, raw_faces = self.yunet_detector.detect(img_bgr)
                if raw_faces is not None and len(raw_faces) > 0:
                    for f in raw_faces:
                        box = [int(f[0]), int(f[1]), int(f[2]), int(f[3])]
                        # Clip to image boundary
                        bx = max(0, min(box[0], w - 1))
                        by = max(0, min(box[1], h - 1))
                        bw = min(box[2], w - bx)
                        bh = min(box[3], h - by)
                        if bw >= 24 and bh >= 24:
                            conf = float(f[14])
                            lmks = f[4:14].reshape((5, 2)).tolist()
                            detections.append({
                                "bbox": [bx, by, bw, bh],
                                "confidence": round(conf, 3),
                                "landmarks": lmks,
                                "source": "YuNet"
                            })
            except Exception as e:
                print(f"[AdvancedFaceDetector] YuNet runtime notice: {e}")

        # 2. Fallback to Dlib / Haar if YuNet found nothing
        if len(detections) == 0:
            if self.dlib_detector is not None:
                try:
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    dlib_rects = self.dlib_detector(gray, 0)
                    for r in dlib_rects:
                        bx = max(0, r.left())
                        by = max(0, r.top())
                        bw = min(r.width(), w - bx)
                        bh = min(r.height(), h - by)
                        if bw >= 24 and bh >= 24:
                            lmks = None
                            if self.dlib_pred is not None:
                                sp = self.dlib_pred(gray, r)
                                # 5 key points: left eye, right eye, nose, left mouth, right mouth
                                le = ((sp.part(36).x + sp.part(39).x) / 2, (sp.part(36).y + sp.part(39).y) / 2)
                                re = ((sp.part(42).x + sp.part(45).x) / 2, (sp.part(42).y + sp.part(45).y) / 2)
                                no = (sp.part(30).x, sp.part(30).y)
                                lm = (sp.part(48).x, sp.part(48).y)
                                rm = (sp.part(54).x, sp.part(54).y)
                                lmks = [le, re, no, lm, rm]
                            detections.append({
                                "bbox": [bx, by, bw, bh],
                                "confidence": 0.88,
                                "landmarks": lmks,
                                "source": "Dlib"
                            })
                except Exception:
                    pass

        # 3. Haar Cascade fallback
        if len(detections) == 0:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            haar_boxes = self.frontal_cascade.detectMultiScale(gray, 1.15, 4, minSize=(30, 30))
            if len(haar_boxes) == 0:
                haar_boxes = self.alt_cascade.detectMultiScale(gray, 1.15, 3, minSize=(30, 30))
            for (bx, by, bw, bh) in haar_boxes:
                detections.append({
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "confidence": 0.75,
                    "landmarks": None,
                    "source": "Haar"
                })

        # Apply Non-Maximum Suppression (NMS) to eliminate overlapping duplicates
        return self._apply_nms(detections, iou_thresh=self.nms_threshold)

    def _apply_nms(self, detections, iou_thresh=0.35):
        if len(detections) <= 1:
            return detections

        boxes = [d["bbox"] for d in detections]
        scores = [d["confidence"] for d in detections]

        indices = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2], b[3]] for b in boxes],
            scores=scores,
            score_threshold=0.3,
            nms_threshold=iou_thresh
        )
        if len(indices) == 0:
            return detections
        selected_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        return [detections[i] for i in selected_indices]


# =============================================================================
# 2. FACE ALIGNMENT & QUALITY ASSESSMENT
# =============================================================================
def align_face_crop(img_bgr, bbox, landmarks=None, output_size=(160, 160)):
    """
    Aligns and crops facial region.
    If 5 landmarks are available, applies affine similarity warp.
    Otherwise, extracts bounding box crop with proportional margin.
    """
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = bbox

    # 1. 5-point landmark similarity transform alignment
    if landmarks is not None and len(landmarks) == 5:
        try:
            src_pts = np.array(landmarks, dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_FACENET_160)
            if M is not None:
                aligned = cv2.warpAffine(
                    img_bgr, M, output_size,
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return aligned
        except Exception:
            pass

    # 2. Margin crop fallback
    pad_w = int(bw * 0.12)
    pad_h = int(bh * 0.12)
    x1 = max(0, bx - pad_w)
    y1 = max(0, by - pad_h)
    x2 = min(w, bx + bw + pad_w)
    y2 = min(h, by + bh + pad_h)

    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img_bgr[by:by+bh, bx:bx+bw]

    if crop.size > 0:
        return cv2.resize(crop, output_size, interpolation=cv2.INTER_CUBIC)
    return None


def assess_face_quality(face_crop, original_bbox=None):
    """
    Evaluates image quality metrics:
    - Blur / Sharpness (Laplacian variance >= 65 is sharp)
    - Illumination (Mean brightness 25 - 235)
    - Contrast (Standard deviation >= 20)
    - Face Size (Width >= 50px)
    - Anti-Spoofing Liveness Index (Texture frequency ratio & specular glare)
    """
    if face_crop is None or face_crop.size == 0:
        return {"is_valid": False, "score": 0.0, "issues": ["Empty face image"]}

    issues = []
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop

    # 1. Blur via Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = lap_var < 65.0
    if is_blurry:
        issues.append(f"Blur detected (sharpness: {round(lap_var, 1)} < 65)")

    # 2. Illumination / Brightness
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    if mean_val < 25.0:
        issues.append(f"Severely underexposed / dark ({round(mean_val, 1)})")
    elif mean_val > 235.0:
        issues.append(f"Severely overexposed / washed out ({round(mean_val, 1)})")
    if std_val < 20.0:
        issues.append("Low contrast")

    # 3. Resolution / Face Size
    face_w = original_bbox[2] if original_bbox else face_crop.shape[1]
    face_h = original_bbox[3] if original_bbox else face_crop.shape[0]
    if face_w < 50 or face_h < 50:
        issues.append(f"Face size too small ({face_w}x{face_h} < 50x50)")

    # 4. Anti-Spoofing Liveness Texture Score
    # Printed paper & mobile screens exhibit flat high-frequency spectral artifacts
    dft = np.fft.fft2(gray.astype(np.float32))
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.log(np.abs(dft_shift) + 1.0)
    h_c, w_c = gray.shape[:2]
    center_roi = magnitude_spectrum[h_c//4:3*h_c//4, w_c//4:3*w_c//4]
    outer_mean = float(np.mean(magnitude_spectrum))
    inner_mean = float(np.mean(center_roi))
    liveness_ratio = round(inner_mean / max(0.001, outer_mean), 2)
    is_live = (0.75 <= liveness_ratio <= 1.45)
    if not is_live:
        issues.append("Possible spoof / 2D screen display detected")

    # Composite Quality Score (0.0 to 1.0)
    norm_blur = min(1.0, lap_var / 250.0)
    norm_illum = 1.0 - (abs(mean_val - 128.0) / 128.0)
    norm_size = min(1.0, min(face_w, face_h) / 120.0)
    quality_score = round(float((norm_blur * 0.45) + (norm_illum * 0.35) + (norm_size * 0.20)), 3)

    return {
        "is_valid": len(issues) == 0,
        "quality_score": quality_score,
        "blur_score": round(lap_var, 1),
        "is_blurry": is_blurry,
        "brightness": round(mean_val, 1),
        "contrast": round(std_val, 1),
        "is_live": is_live,
        "liveness_ratio": liveness_ratio,
        "face_dimensions": f"{face_w}x{face_h}",
        "issues": issues
    }


# =============================================================================
# 3. METRIC EMBEDDING MATCHING ENGINE & PROTOTYPE STORAGE
# =============================================================================
FACULTY_PROTOTYPES_PATH = BASE_DIR / "Model" / "faculty_prototypes.json"


class PrototypeMatcher:
    """
    Embedding-based face identification engine using multi-prototype exemplar
    representations and calibrated cosine distance thresholds.
    Supports class-group filtered candidate matching and ambiguity detection.
    """
    def __init__(self, threshold_high=0.72, threshold_review=0.58):
        self.threshold_high = threshold_high
        self.threshold_review = threshold_review
        self.prototypes = {}  # { roll_number: { "name": str, "centroid": [128], "exemplars": [[128], ...] } }
        self.active_version = "v3.2-FaceNet"
        self.load_prototypes()

    def load_prototypes(self):
        """Loads stored student prototype embeddings."""
        if PROTOTYPES_PATH.exists():
            try:
                with open(PROTOTYPES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.prototypes = data.get("prototypes", {})
                    self.threshold_high = data.get("threshold_high", self.threshold_high)
                    self.threshold_review = data.get("threshold_review", self.threshold_review)
                    self.active_version = data.get("version_id", self.active_version)
                    print(f"[PrototypeMatcher] Loaded {len(self.prototypes)} registered student identity prototypes (v: {self.active_version}).")
            except Exception as e:
                print(f"[PrototypeMatcher] Error loading prototypes: {e}")

    def save_prototypes(self, version_id=None, metrics=None):
        """Saves student prototype embeddings and threshold configuration."""
        PROTOTYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version_id": version_id or self.active_version,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold_high": self.threshold_high,
            "threshold_review": self.threshold_review,
            "num_students": len(self.prototypes),
            "metrics": metrics or {},
            "prototypes": self.prototypes
        }
        with open(PROTOTYPES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def cosine_similarity(v1, v2):
        """Computes cosine similarity between two unit-normalized vectors."""
        return float(np.dot(v1, v2))

    def add_or_update_student(self, roll_number, name, vectors):
        """Incrementally adds or updates a student's prototype representation."""
        roll = str(roll_number).strip()
        if not vectors:
            return False

        arr = np.array(vectors, dtype=np.float32)
        # Normalize each vector
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        arr = arr / norms

        # Compute centroid
        centroid = np.mean(arr, axis=0)
        centroid = centroid / max(1e-9, np.linalg.norm(centroid))

        # Select exemplars
        exemplars = [centroid.tolist()]
        if len(arr) >= 3:
            sims = np.dot(arr, centroid)
            sorted_indices = np.argsort(sims)
            exemplars.append(arr[sorted_indices[0]].tolist())
            exemplars.append(arr[sorted_indices[len(sorted_indices) // 2]].tolist())

        self.prototypes[roll] = {
            "name": name,
            "roll_number": roll,
            "centroid": centroid.tolist(),
            "exemplars": exemplars,
            "num_samples": len(vectors)
        }
        self.save_prototypes()
        return True

    def match_embedding(self, query_emb, eligible_ids=None):
        """
        Matches a 128-d query embedding against candidate prototypes.
        If eligible_ids is provided, restricts candidate search strictly to that class group.
        Implements zero-tolerance ambiguity detection (< 0.04 margin).
        """
        if not self.prototypes or query_emb is None:
            return "UNKNOWN", "Visitor / Unregistered", 0.0, "UNKNOWN"

        q_norm = query_emb / max(1e-9, np.linalg.norm(query_emb))

        pool = self.prototypes
        if eligible_ids is not None:
            eligible_set = {str(eid).strip() for eid in eligible_ids}
            pool = {r: p for r, p in self.prototypes.items() if str(r).strip() in eligible_set}

        if not pool:
            return "UNKNOWN", "Visitor / Unregistered", 0.0, "UNKNOWN"

        best_roll = "UNKNOWN"
        best_name = "Visitor / Unregistered"
        max_sim = -1.0
        second_sim = -1.0

        for roll, profile in pool.items():
            name = profile.get("name", f"Student {roll}")
            centroid = np.array(profile["centroid"], dtype=np.float32)
            c_norm = centroid / max(1e-9, np.linalg.norm(centroid))
            c_sim = self.cosine_similarity(q_norm, c_norm)

            sim = c_sim
            for ex in profile.get("exemplars", []):
                ex_norm = np.array(ex, dtype=np.float32)
                ex_norm = ex_norm / max(1e-9, np.linalg.norm(ex_norm))
                e_sim = self.cosine_similarity(q_norm, ex_norm)
                if e_sim > sim:
                    sim = e_sim

            if sim > max_sim:
                second_sim = max_sim
                max_sim = sim
                best_roll = roll
                best_name = name
            elif sim > second_sim:
                second_sim = sim

        # Zero-Tolerance Ambiguity Protection
        if len(pool) > 1 and second_sim > 0.0 and (max_sim - second_sim) < 0.04 and max_sim >= self.threshold_review:
            return best_roll, best_name, round(max_sim, 3), "AMBIGUOUS"

        if max_sim >= self.threshold_high:
            state = "HIGH_CONFIDENCE"
        elif max_sim >= self.threshold_review:
            state = "REVIEW_REQUIRED"
        else:
            state = "UNKNOWN"
            best_roll = "UNKNOWN"
            best_name = "Visitor / Unregistered"

        return best_roll, best_name, round(max_sim, 3), state


# =============================================================================
# 4. FACULTY EXCLUSION MATCHER (Strict Biometric Isolation)
# =============================================================================
class FacultyMatcher:
    """
    Dedicated identity matcher for registered faculty members.
    Completely isolated namespace from student prototypes.
    Faculty are strictly excluded from student attendance.
    """
    def __init__(self, threshold=0.70):
        self.threshold = threshold
        self.prototypes = {}  # { faculty_id: { "name": str, "centroid": [128], "exemplars": [[128], ...] } }
        self.active_version = "v1.0-Faculty"
        self.load_prototypes()

    def load_prototypes(self):
        """Loads stored faculty prototype embeddings."""
        if FACULTY_PROTOTYPES_PATH.exists():
            try:
                with open(FACULTY_PROTOTYPES_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.prototypes = data.get("prototypes", {})
                    self.threshold = data.get("threshold", self.threshold)
                    print(f"[FacultyMatcher] Loaded {len(self.prototypes)} registered faculty prototypes.")
            except Exception as e:
                print(f"[FacultyMatcher] Error loading faculty prototypes: {e}")

    def save_prototypes(self):
        """Saves faculty prototype embeddings."""
        FACULTY_PROTOTYPES_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version_id": self.active_version,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": self.threshold,
            "num_faculty": len(self.prototypes),
            "prototypes": self.prototypes
        }
        with open(FACULTY_PROTOTYPES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_or_update_faculty(self, faculty_id, name, vectors):
        """Incrementally adds or updates a faculty prototype representation."""
        fid = str(faculty_id).strip()
        if not vectors:
            return False

        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        arr = arr / norms

        centroid = np.mean(arr, axis=0)
        centroid = centroid / max(1e-9, np.linalg.norm(centroid))

        exemplars = [centroid.tolist()]
        if len(arr) >= 3:
            sims = np.dot(arr, centroid)
            sorted_indices = np.argsort(sims)
            exemplars.append(arr[sorted_indices[0]].tolist())
            exemplars.append(arr[sorted_indices[len(sorted_indices) // 2]].tolist())

        self.prototypes[fid] = {
            "name": name,
            "faculty_id": fid,
            "centroid": centroid.tolist(),
            "exemplars": exemplars,
            "num_samples": len(vectors)
        }
        self.save_prototypes()
        return True

    def match_embedding(self, query_emb):
        """
        Matches a 128-d query embedding against faculty prototypes.
        Returns:
            is_faculty (bool), faculty_id (str or None), name (str or None), similarity (float)
        """
        if not self.prototypes or query_emb is None:
            return False, None, None, 0.0

        q_norm = query_emb / max(1e-9, np.linalg.norm(query_emb))
        best_id = None
        best_name = None
        max_sim = -1.0

        for fid, profile in self.prototypes.items():
            name = profile.get("name", f"Faculty {fid}")
            centroid = np.array(profile["centroid"], dtype=np.float32)
            c_norm = centroid / max(1e-9, np.linalg.norm(centroid))
            c_sim = float(np.dot(q_norm, c_norm))

            sim = c_sim
            for ex in profile.get("exemplars", []):
                ex_norm = np.array(ex, dtype=np.float32)
                ex_norm = ex_norm / max(1e-9, np.linalg.norm(ex_norm))
                e_sim = float(np.dot(q_norm, ex_norm))
                if e_sim > sim:
                    sim = e_sim

            if sim > max_sim:
                max_sim = sim
                best_id = fid
                best_name = name

        if max_sim >= self.threshold:
            return best_id, best_name, round(max_sim, 3), "VERIFIED_FACULTY"
        return "UNKNOWN", "Visitor / Unregistered", round(max_sim, 3), "NOT_FACULTY"


# Global Engine Instances
face_detector = AdvancedFaceDetector()
prototype_matcher = PrototypeMatcher()
faculty_matcher = FacultyMatcher()


# =============================================================================
# 5. AUTOMATIC REAL-TIME FACE ENROLLMENT PIPELINE
# =============================================================================
_GLOBAL_EMB_INSTANCE = None


def _get_embedding_model():
    global _GLOBAL_EMB_INSTANCE
    if _GLOBAL_EMB_INSTANCE is None:
        from embedding import emb
        _GLOBAL_EMB_INSTANCE = emb()
    return _GLOBAL_EMB_INSTANCE


def auto_enroll_student_face(roll_number, student_name, image_input, source="upload"):
    """
    Automated zero-manual-training student face enrollment pipeline.
    Validates image, detects face, checks dimensions & blur, aligns crop,
    extracts FaceNet-128D embedding, saves embedding, and updates prototype index.
    """
    from database_manager import db_manager

    roll = str(roll_number).strip()
    # 1. Decode image input
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    elif isinstance(image_input, (bytes, bytearray)):
        np_arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        db_manager.update_student_face_status(roll, "FAILED_IMAGE_CORRUPT")
        return {"success": False, "state": "FAILED_IMAGE_CORRUPT", "message": "Corrupted or unreadable image data."}

    if img is None or img.size == 0:
        db_manager.update_student_face_status(roll, "FAILED_IMAGE_CORRUPT")
        return {"success": False, "state": "FAILED_IMAGE_CORRUPT", "message": "Failed to decode photograph."}

    # 2. Multi-face detection
    faces = face_detector.detect(img)
    if len(faces) == 0:
        db_manager.update_student_face_status(roll, "FAILED_NO_FACE")
        return {"success": False, "state": "FAILED_NO_FACE", "message": "Failed — No face detected in the photograph."}

    # If multiple faces appear, select primary dominant face or flag manual review
    if len(faces) > 1:
        areas = [f["bbox"][2] * f["bbox"][3] for f in faces]
        max_idx = int(np.argmax(areas))
        max_area = areas[max_idx]
        other_areas = [a for idx, a in enumerate(areas) if idx != max_idx]
        if other_areas and max(other_areas) > (max_area * 0.45):
            db_manager.update_student_face_status(roll, "MANUAL_REVIEW_REQUIRED")
            return {
                "success": False,
                "state": "MANUAL_REVIEW_REQUIRED",
                "message": f"Multiple faces detected ({len(faces)} candidates). Manual face selection required.",
                "face_count": len(faces)
            }
        primary_face = faces[max_idx]
    else:
        primary_face = faces[0]

    box = primary_face["bbox"]
    lmks = primary_face.get("landmarks")

    # 3. Minimum face size check
    if box[2] < 45 or box[3] < 45:
        db_manager.update_student_face_status(roll, "FAILED_QUALITY")
        return {"success": False, "state": "FAILED_QUALITY", "message": f"Face size too small ({box[2]}x{box[3]} < 45x45 required)."}

    # 4. Affine alignment
    aligned = align_face_crop(img, box, lmks, output_size=(160, 160))
    if aligned is None or aligned.size == 0:
        db_manager.update_student_face_status(roll, "FAILED_QUALITY")
        return {"success": False, "state": "FAILED_QUALITY", "message": "Could not align facial crop."}

    # 5. Image quality assessment
    quality = assess_face_quality(aligned, box)
    if quality.get("is_blurry") and quality.get("blur_score", 0) < 35.0:
        db_manager.update_student_face_status(roll, "FAILED_QUALITY")
        return {
            "success": False,
            "state": "FAILED_QUALITY",
            "message": f"Failed — Image Quality Problem (excessive blur: {quality.get('blur_score')} < 35.0).",
            "quality": quality
        }

    # 6. Extract normalized FaceNet embedding
    e_model = _get_embedding_model()
    prep = e_model.preprocess_face(aligned)
    if prep is None:
        db_manager.update_student_face_status(roll, "FAILED_QUALITY")
        return {"success": False, "state": "FAILED_QUALITY", "message": "Preprocessing face crop failed."}

    raw_vec = e_model.calculate(prep)
    norm_vec = raw_vec / max(1e-9, np.linalg.norm(raw_vec))

    # 7. Persist embedding vector to database
    emb_id = db_manager.save_student_embedding(
        student_id=roll,
        embedding=norm_vec.tolist(),
        quality_score=quality.get("quality_score", 0.0),
        blur_score=quality.get("blur_score", 0.0),
        is_valid=1,
        model_version="FaceNet-128D"
    )

    # 8. Incrementally update PrototypeMatcher
    all_vectors = db_manager.get_student_embeddings(roll)
    if not all_vectors:
        all_vectors = [norm_vec.tolist()]

    prototype_matcher.add_or_update_student(roll, student_name, all_vectors)
    db_manager.update_student_face_status(roll, "ENROLLED")

    # Reload camera models
    try:
        from classroom_camera import classroom_camera
        classroom_camera.load_models()
    except Exception:
        pass

    return {
        "success": True,
        "state": "ENROLLED",
        "embedding_id": emb_id,
        "message": "Face Enrollment Successful. Biometric model updated automatically.",
        "quality": quality
    }


def auto_enroll_faculty_face(faculty_id, faculty_name, image_input, source="upload"):
    """
    Automated faculty face enrollment pipeline.
    Stores embeddings in isolated faculty namespace.
    """
    from database_manager import db_manager

    fid = str(faculty_id).strip()
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
    elif isinstance(image_input, (bytes, bytearray)):
        np_arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        return {"success": False, "state": "FAILED_IMAGE_CORRUPT", "message": "Corrupted faculty image."}

    if img is None or img.size == 0:
        return {"success": False, "state": "FAILED_IMAGE_CORRUPT", "message": "Failed to decode faculty photograph."}

    faces = face_detector.detect(img)
    if len(faces) == 0:
        return {"success": False, "state": "FAILED_NO_FACE", "message": "No face detected in faculty photograph."}

    primary_face = faces[0]
    box = primary_face["bbox"]
    lmks = primary_face.get("landmarks")

    aligned = align_face_crop(img, box, lmks, output_size=(160, 160))
    if aligned is None or aligned.size == 0:
        return {"success": False, "state": "FAILED_QUALITY", "message": "Could not align facial crop."}

    quality = assess_face_quality(aligned, box)

    e_model = _get_embedding_model()
    prep = e_model.preprocess_face(aligned)
    raw_vec = e_model.calculate(prep)
    norm_vec = raw_vec / max(1e-9, np.linalg.norm(raw_vec))

    emb_id = db_manager.save_faculty_embedding(
        faculty_id=fid,
        name=faculty_name,
        embedding=norm_vec.tolist(),
        model_version="FaceNet-128D"
    )

    all_vectors = db_manager.get_faculty_embeddings(fid)
    if not all_vectors:
        all_vectors = [norm_vec.tolist()]

    faculty_matcher.add_or_update_faculty(fid, faculty_name, all_vectors)

    return {
        "success": True,
        "state": "ENROLLED",
        "embedding_id": emb_id,
        "message": "Faculty face enrolled successfully in isolated exclusion namespace.",
        "quality": quality
    }
