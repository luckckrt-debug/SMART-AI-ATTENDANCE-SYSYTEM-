import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class face:
    def __init__(self):
        # Load primary cascade
        xml_path = BASE_DIR / "FaceDetection" / "faces.xml"
        if xml_path.exists():
            self.cascade = cv2.CascadeClassifier(str(xml_path))
        else:
            self.cascade = cv2.CascadeClassifier()

        if self.cascade.empty():
            default_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(default_path)
            print(f"[FaceDetection] Using OpenCV default Haar cascade: {default_path}")

        # Fallback cascade
        self.alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

        self.x = None
        self.y = None
        self.w = None
        self.h = None

    def detectFace(self, img):
        if img is None or img.size == 0:
            return [], []

        h, w = img.shape[:2]
        # Fast downscale if image is high-resolution
        scale = 1.0
        if max(h, w) > 1000:
            scale = 1000.0 / max(h, w)
            proc_img = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            proc_img = img

        if len(proc_img.shape) == 3:
            grey = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        else:
            grey = proc_img

        # Equalize histogram for illumination robustness
        grey = cv2.equalizeHist(grey)

        faces = self.cascade.detectMultiScale(
            grey,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(30, 30)
        )

        # Fallback to alternate cascade if primary detected no faces
        if len(faces) == 0 and not self.alt_cascade.empty():
            faces = self.alt_cascade.detectMultiScale(
                grey,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

        cropped = []
        coor = []
        for (x, y, fw, fh) in faces:
            # Scale coordinates back to original image
            ox = int(x / scale)
            oy = int(y / scale)
            ofw = int(fw / scale)
            ofh = int(fh / scale)
            # Clip bounds
            ox = max(0, min(ox, w - 1))
            oy = max(0, min(oy, h - 1))
            ofw = min(ofw, w - ox)
            ofh = min(ofh, h - oy)

            if ofw > 20 and ofh > 20:
                cropped.append(img[oy:oy+ofh, ox:ox+ofw])
                coor.append([ox, oy, ofw, ofh])

        return cropped, coor
