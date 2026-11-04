import os
import json
import warnings
from pathlib import Path
import numpy as np
import h5py
import cv2

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
FACENET_PATH = BASE_DIR / "PreTrained_model" / "facenet_keras.h5"

_GLOBAL_FACENET_MODEL = None


def _load_facenet_model():
    """
    Loads facenet_keras.h5 handling Python 3 cross-version Lambda bytecode serialization.
    Caches the loaded model globally to ensure high inference speed.
    """
    global _GLOBAL_FACENET_MODEL
    if _GLOBAL_FACENET_MODEL is not None:
        return _GLOBAL_FACENET_MODEL

    if not FACENET_PATH.exists():
        print(f"[FaceNet Warning] Pretrained FaceNet model not found at {FACENET_PATH}.")
        return None

    try:
        import keras.src.utils.generic_utils as gu
        import keras.src.saving.serialization_lib as sl
        from keras.models import model_from_json

        # Enable unsafe deserialization for trusted local model weights
        sl.enable_unsafe_deserialization()

        def scale_sum(inputs, scale=1.0):
            return inputs[0] + inputs[1] * scale

        dumped_func = gu.func_dump(scale_sum)[0]

        with h5py.File(str(FACENET_PATH), "r") as f:
            raw_cfg = f.attrs.get("model_config")
            if isinstance(raw_cfg, bytes):
                raw_cfg = raw_cfg.decode("utf-8")
            config = json.loads(raw_cfg)
            for layer in config["config"]["layers"]:
                if layer["class_name"] == "Lambda":
                    layer["config"]["function"] = dumped_func

        model = model_from_json(json.dumps(config))
        model.load_weights(str(FACENET_PATH))
        print("[FaceNet] Model loaded successfully.")
        _GLOBAL_FACENET_MODEL = model
        return _GLOBAL_FACENET_MODEL
    except Exception as e:
        print(f"[FaceNet] Standard load failed ({e}), attempting fallback load_model...")
        from keras.models import load_model
        model = load_model(str(FACENET_PATH), compile=False)
        _GLOBAL_FACENET_MODEL = model
        return _GLOBAL_FACENET_MODEL


class emb:
    def __init__(self):
        self.model = _load_facenet_model()

    @staticmethod
    def preprocess_face(face_bgr):
        """
        Resize face to (160, 160), convert to float, and format for FaceNet inference.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (160, 160))
        # Standard normalization: scale pixels to [0, 1] or standard standardization
        float_img = resized.astype("float32") / 255.0
        return np.expand_dims(float_img, axis=0)

    def calculate(self, img):
        """
        Computes 128-dimensional embedding from normalized (1, 160, 160, 3) image.
        """
        if self.model is None:
            raise RuntimeError("FaceNet model is not loaded. Please ensure facenet_keras.h5 is in PreTrained_model/.")
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        embeddings = self.model.predict(img, verbose=0)
        return embeddings[0]
