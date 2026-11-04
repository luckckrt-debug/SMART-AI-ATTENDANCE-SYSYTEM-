"""
Advanced Dataset Management Engine
Handles:
1. Dataset Integrity Audits (Corrupt images, Missing faces, Multiple-face group photos)
2. Quality Assessment (Laplacian blur, Illumination, Face dimensions)
3. Duplicate Image Detection (dHash / perceptual hashing)
4. Group Photo Multi-Face Extractor (Crops individual faces for admin assignment)
5. Class Distribution & Training Readiness Reports
"""

import os
import json
import base64
from pathlib import Path
import cv2
import numpy as np

from ai_engine import face_detector, align_face_crop, assess_face_quality, prototype_matcher
from database_manager import db_manager

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_DIR = BASE_DIR / "people"
PROFILES_DIR = BASE_DIR / "static" / "profiles"


def calculate_dhash(image, hash_size=8):
    """Computes difference hash (dHash) to detect identical or near-duplicate images."""
    if image is None or image.size == 0:
        return ""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:]


class DatasetManager:
    """
    Production dataset management and quality validation engine.
    """
    def __init__(self):
        PEOPLE_DIR.mkdir(parents=True, exist_ok=True)
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def audit_dataset(self):
        """
        Conducts a deep audit of the student training dataset in people/.
        Returns comprehensive metrics, image-level quality reports, and warnings.
        """
        students = db_manager.get_all_students()
        student_map = {str(s["roll_number"]): s for s in students}

        report = {
            "total_registered_students": len(students),
            "students_with_photos": 0,
            "total_images": 0,
            "total_faces_detected": 0,
            "high_quality_images": 0,
            "blurry_images": 0,
            "multiple_faces_images": 0,
            "no_face_images": 0,
            "duplicate_images_found": 0,
            "student_summaries": [],
            "image_details": [],
            "flagged_issues": []
        }

        seen_hashes = {}  # {hash: (roll, filename)}

        if not PEOPLE_DIR.exists():
            return report

        folders = sorted([f for f in os.listdir(PEOPLE_DIR) if (PEOPLE_DIR / f).is_dir()])

        for folder in folders:
            folder_path = PEOPLE_DIR / folder
            # Parse roll number
            import re
            m = re.match(r"^(\d+)[_\s-]*(.*)$", folder)
            roll = m.group(1) if m else folder
            raw_name = m.group(2) if m else folder

            student = student_map.get(str(roll), {"name": raw_name, "roll_number": roll, "department": "", "section": ""})
            img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            if len(img_files) > 0:
                report["students_with_photos"] += 1

            student_summary = {
                "roll_number": roll,
                "name": student.get("name", raw_name),
                "folder": folder,
                "image_count": len(img_files),
                "valid_faces_count": 0,
                "has_group_photos": False,
                "has_blurry": False,
                "has_duplicates": False,
                "status": "Ready" if len(img_files) >= 3 else ("Needs More Photos" if len(img_files) > 0 else "No Images")
            }

            for img_name in img_files:
                img_path = folder_path / img_name
                report["total_images"] += 1

                img = cv2.imread(str(img_path))
                if img is None:
                    report["flagged_issues"].append({
                        "roll": roll,
                        "file": img_name,
                        "issue": "Corrupt or unreadable image file"
                    })
                    continue

                # Duplicate detection via dHash
                h_val = calculate_dhash(img)
                if h_val in seen_hashes:
                    prev_roll, prev_file = seen_hashes[h_val]
                    report["duplicate_images_found"] += 1
                    student_summary["has_duplicates"] = True
                    report["flagged_issues"].append({
                        "roll": roll,
                        "file": img_name,
                        "issue": f"Duplicate of image in {prev_roll}/{prev_file}"
                    })
                else:
                    seen_hashes[h_val] = (roll, img_name)

                # Face detection
                faces = face_detector.detect(img)
                num_faces = len(faces)
                report["total_faces_detected"] += num_faces

                if num_faces == 0:
                    report["no_face_images"] += 1
                    report["flagged_issues"].append({
                        "roll": roll,
                        "file": img_name,
                        "issue": "No face detected in photo"
                    })
                elif num_faces > 1:
                    report["multiple_faces_images"] += 1
                    student_summary["has_group_photos"] = True
                    report["flagged_issues"].append({
                        "roll": roll,
                        "file": img_name,
                        "issue": f"Group photo detected ({num_faces} faces in one training sample)"
                    })

                # Quality assessment on primary face or image
                primary_bbox = faces[0]["bbox"] if num_faces > 0 else [0, 0, img.shape[1], img.shape[0]]
                crop = align_face_crop(img, primary_bbox, faces[0]["landmarks"] if num_faces > 0 else None)
                quality = assess_face_quality(crop if crop is not None else img, primary_bbox)

                if quality["is_blurry"]:
                    report["blurry_images"] += 1
                    student_summary["has_blurry"] = True

                if quality["is_valid"] and num_faces == 1:
                    report["high_quality_images"] += 1
                    student_summary["valid_faces_count"] += 1

                report["image_details"].append({
                    "roll": roll,
                    "student_name": student.get("name", raw_name),
                    "filename": img_name,
                    "faces_count": num_faces,
                    "quality_score": quality["quality_score"],
                    "blur_score": quality["blur_score"],
                    "is_blurry": quality["is_blurry"],
                    "is_valid": quality["is_valid"],
                    "issues": quality["issues"]
                })

            report["student_summaries"].append(student_summary)

        return report

    def extract_faces_from_group_photo(self, img_bytes):
        """
        Extracts all faces from an uploaded group photograph.
        Returns individual cropped face thumbnails and suggested student matches
        for administrator review and confirmation.
        """
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode uploaded group photograph.")

        faces = face_detector.detect(img)
        crops_data = []

        # Load FaceNet embedding extractor for candidate suggestions
        from embedding import emb
        e_model = emb()

        for idx, f in enumerate(faces):
            box = f["bbox"]
            lmks = f["landmarks"]
            aligned = align_face_crop(img, box, lmks, output_size=(160, 160))
            if aligned is None or aligned.size == 0:
                continue

            quality = assess_face_quality(aligned, box)

            # Generate suggested identity using FaceNet prototype matcher
            suggested_roll = "UNKNOWN"
            suggested_name = "Unassigned"
            match_conf = 0.0

            try:
                prep = e_model.preprocess_face(aligned)
                if prep is not None:
                    vec = e_model.calculate(prep)
                    s_roll, s_name, sim, state = prototype_matcher.match_embedding(vec)
                    if state in ["HIGH_CONFIDENCE", "REVIEW_REQUIRED"]:
                        suggested_roll = s_roll
                        suggested_name = s_name
                        match_conf = sim
            except Exception:
                pass

            # Convert aligned crop to base64 JPEG
            _, buf = cv2.imencode(".jpg", aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64_crop = base64.b64encode(buf).decode("utf-8")

            crops_data.append({
                "face_index": idx + 1,
                "bbox": box,
                "thumbnail": f"data:image/jpeg;base64,{b64_crop}",
                "quality": quality,
                "suggested_roll": suggested_roll,
                "suggested_name": suggested_name,
                "confidence": round(match_conf, 3)
            })

        return {
            "total_faces_found": len(crops_data),
            "faces": crops_data
        }

    def assign_crop_to_student(self, roll_number, crop_b64):
        """
        Assigns a verified face crop to a student's training dataset in people/.
        """
        if "," in crop_b64:
            crop_b64 = crop_b64.split(",", 1)[1]

        raw_bytes = base64.b64decode(crop_b64)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Invalid cropped image payload.")

        student = db_manager.get_student(roll_number)
        name = student["name"] if student else f"Student_{roll_number}"
        safe_name = name.replace(" ", "_").upper()
        folder_name = f"{roll_number}_{safe_name}"

        student_dir = PEOPLE_DIR / folder_name
        student_dir.mkdir(parents=True, exist_ok=True)

        existing = [f for f in os.listdir(student_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        next_idx = len(existing) + 1
        target_path = student_dir / f"{next_idx}.jpg"

        cv2.imwrite(str(target_path), img)

        # Update profile photo if not set
        profile_dest = PROFILES_DIR / f"{roll_number}.jpg"
        if not profile_dest.exists():
            cv2.imwrite(str(profile_dest), img)
            db_manager.update_student(roll_number, {"profile_photo": f"/static/profiles/{roll_number}.jpg"})

        return True, f"Successfully assigned face sample #{next_idx} to {name} (Roll: {roll_number})."

    def delete_image(self, roll_number, filename):
        """Safely removes a poor-quality or incorrect image from a student's folder."""
        # Find folder
        for folder in os.listdir(PEOPLE_DIR):
            if folder.startswith(str(roll_number)):
                target = PEOPLE_DIR / folder / filename
                if target.exists():
                    target.unlink()
                    return True, f"Removed {filename} from {folder}."
        return False, "Target file not found."


dataset_manager = DatasetManager()
