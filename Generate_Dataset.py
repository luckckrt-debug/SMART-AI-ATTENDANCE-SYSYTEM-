import os
import re
import cv2
import numpy as np
import base64
from pathlib import Path
from database_manager import db_manager
from FaceDetection.face_detection import face

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_DIR = BASE_DIR / "people"


def sanitize_filename(name):
    return re.sub(r"[^\w\-_]", "", str(name).strip().replace(" ", "_"))


def Generate_Data(name, roll_no, max_pics=30, camera_index=0):
    """
    Captures facial dataset via local OpenCV webcam and enrolls student.
    Returns: (success: bool, message: str, num_captured: int)
    """
    clean_name = sanitize_filename(name)
    clean_roll = sanitize_filename(roll_no)

    if not clean_name or not clean_roll:
        return False, "Student name and roll number cannot be empty.", 0

    # Enroll in database first
    enrolled, msg = db_manager.enroll_student(clean_roll, clean_name)
    if not enrolled:
        return False, msg, 0

    student_folder = PEOPLE_DIR / f"{clean_roll}_{clean_name}"
    student_folder.mkdir(parents=True, exist_ok=True)

    fd = face()
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return False, f"Could not open camera device at index {camera_index}.", 0

    pic_no = 0
    try:
        while pic_no < max_pics:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            cropped_faces, coords = fd.detectFace(frame)

            for cropped in cropped_faces:
                if cropped is not None and cropped.size > 0:
                    pic_no += 1
                    img_path = student_folder / f"{pic_no}.jpg"
                    cv2.imwrite(str(img_path), cropped)

                    if pic_no >= max_pics:
                        break

            # Draw preview rectangle if window can be displayed
            for (x, y, w, h) in coords:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            try:
                cv2.imshow("Enrolling Student - Press 'q' to Quit", frame)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break
            except Exception:
                # Running headless / web environment where GUI display is unavailable
                pass

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return True, f"Successfully captured {pic_no} images for {clean_name} (Roll: {clean_roll}).", pic_no


def save_student_captured_images(roll_no, name, base64_images):
    """
    Saves a list of base64-encoded JPEG/PNG images submitted from the web camera UI.
    """
    clean_name = sanitize_filename(name)
    clean_roll = sanitize_filename(roll_no)

    if not clean_name or not clean_roll:
        return False, "Student name and roll number cannot be empty.", 0

    if not base64_images or len(base64_images) == 0:
        return False, "No image frames received from camera.", 0

    # Enroll in database
    enrolled, msg = db_manager.enroll_student(clean_roll, clean_name)
    if not enrolled and "already enrolled" not in msg.lower():
        return False, msg, 0

    student_folder = PEOPLE_DIR / f"{clean_roll}_{clean_name}"
    student_folder.mkdir(parents=True, exist_ok=True)

    fd = face()
    saved_count = 0

    for i, b64_str in enumerate(base64_images):
        try:
            # Strip data URL prefix if present
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                continue

            # Detect and crop face
            crops, _ = fd.detectFace(frame)
            if len(crops) > 0 and crops[0].size > 0:
                target_img = crops[0]
            else:
                target_img = frame

            saved_count += 1
            img_path = student_folder / f"{saved_count}.jpg"
            cv2.imwrite(str(img_path), target_img)
        except Exception as e:
            print(f"[GenerateDataset] Error saving web frame {i}: {e}")

    return True, f"Successfully enrolled {clean_name} ({clean_roll}) with {saved_count} face images.", saved_count
