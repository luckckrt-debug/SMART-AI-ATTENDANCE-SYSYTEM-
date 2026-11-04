"""
Cross-Platform Startup Script for Smart Attendance System
Usage: python run.py
"""
import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Check if running inside virtual environment
VENV_PYTHON = BASE_DIR / "venv" / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
PYVENV_CFG = BASE_DIR / "venv" / "pyvenv.cfg"

# Auto-heal pyvenv.cfg if needed on Windows
if os.name == "nt" and PYVENV_CFG.exists():
    try:
        content = PYVENV_CFG.read_text(encoding="utf-8")
        if "pythoncore" in content.lower() or ("home = " in content and not Path(content.split("home = ")[1].split("\n")[0].strip()).exists()):
            base_py = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python310"
            if base_py.exists():
                PYVENV_CFG.write_text(
                    f"home = {base_py}\ninclude-system-site-packages = false\nversion = 3.10.11\n"
                    f"executable = {base_py / 'python.exe'}\ncommand = {base_py / 'python.exe'} -m venv {BASE_DIR / 'venv'}\n",
                    encoding="utf-8"
                )
    except Exception:
        pass

if VENV_PYTHON.exists() and sys.executable.lower() != str(VENV_PYTHON).lower():
    print(f"[Startup] Switching execution to virtual environment: {VENV_PYTHON}")
    subprocess.run([str(VENV_PYTHON), str(BASE_DIR / "run.py")] + sys.argv[1:])
    sys.exit(0)

# Inside virtual environment or standard environment
print("\n" + "=" * 65)
print("       SMART ATTENDANCE SYSTEM - CROSS-PLATFORM STARTUP")
print("=" * 65)

# 1. Check Python version
print(f"[Python Version] {sys.version.split()[0]} ({sys.executable})")

# 2. Check Database status
try:
    from database_manager import db_manager
    st = db_manager.get_status()
    print(f"[Database] Mode: {st['mode']} | {st['status_message']}")
except Exception as e:
    print(f"[Database Warning] {e}")

# 3. Check Models
facenet_file = BASE_DIR / "PreTrained_model" / "facenet_keras.h5"
landmarks_file = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
cascade_file = BASE_DIR / "FaceDetection" / "faces.xml"

print(f"[Model Check] FaceNet Backbone: {'Found (92MB)' if facenet_file.exists() else 'MISSING'}")
print(f"[Model Check] Dlib Landmarks: {'Found (99MB)' if landmarks_file.exists() else 'MISSING'}")
print(f"[Model Check] Haar Cascade: {'Found' if cascade_file.exists() else 'Using OpenCV Built-in'}")

# 4. Open browser in separate daemon thread
PORT = int(os.getenv("FLASK_RUN_PORT", "5000"))
URL = f"http://127.0.0.1:{PORT}"


def open_browser():
    time.sleep(1.8)
    print(f"[Browser] Opening {URL} in your default browser...")
    try:
        webbrowser.open(URL)
    except Exception as e:
        print(f"[Browser Error] {e}")


import threading
threading.Thread(target=open_browser, daemon=True).start()

# 5. Start Flask app
from app import app
print(f"\n========================================================")
print(f" Web Server running at: {URL}")
print(f" Press CTRL+C to terminate.")
print(f"========================================================\n")

try:
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
except KeyboardInterrupt:
    print("\n[Shutdown] Server stopped by user.")
