import cv2
import dlib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PREDICTOR_PATH = BASE_DIR / "shape_predictor_68_face_landmarks.dat"


class Allign:
    def __init__(self):
        if not PREDICTOR_PATH.exists():
            print(f"[Align Warning] Landmark predictor not found at {PREDICTOR_PATH}.")
            self.pred = None
        else:
            self.pred = dlib.shape_predictor(str(PREDICTOR_PATH))

    def tup(self, left, top, right, bottom):
        return dlib.rectangle(int(left), int(top), int(right), int(bottom))

    def Al(self, faces):
        """
        Detects facial landmarks, calculates roll angle between eyes,
        and aligns the face image via affine rotation.
        """
        if faces is None or faces.size == 0 or self.pred is None:
            return faces, 0.0

        h, w = faces.shape[:2]
        if len(faces.shape) == 3:
            grey = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        else:
            grey = faces

        d = self.tup(0, 0, w, h)
        try:
            features = self.pred(grey, d)
        except Exception as e:
            # If dlib landmark fails on crop, return original
            return faces, 0.0

        # Points 36-41: Left eye, Points 42-47: Right eye
        first_x = int((features.part(36).x + features.part(39).x) / 2)
        first_y = int((features.part(36).y + features.part(39).y) / 2)
        second_x = int((features.part(42).x + features.part(45).x) / 2)
        second_y = int((features.part(42).y + features.part(45).y) / 2)

        dx = second_x - first_x
        dy = second_y - first_y
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image around eyes midpoint to align horizon
        center = ((first_x + second_x) // 2, (first_y + second_y) // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(faces, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return aligned, angle
