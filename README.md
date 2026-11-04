# AI Smart Classroom Attendance System & Session Analytics (v3.0)

A comprehensive, institutional-grade AI Smart Classroom Attendance Management and Session Analytics platform featuring **multi-person facial recognition**, **dynamic session lifecycle management**, **transparent presence duration tracking**, **classroom entry/exit event logging**, **authorized student profile photographs**, and **informed biometric consent safeguards**.

Built with **FaceNet (128-D deep embeddings)**, **Dlib 68-point facial landmark alignment**, **OpenCV**, **Flask**, and a **dual MongoDB & SQLite database engine**.

---

## 🌟 Key Capabilities

### 1. Classroom Session Management
- Configure and manage sessions by **Academic Year**, **Department**, **Class**, **Section**, **Subject**, and **Faculty Name**.
- Automatically generates unique session identifiers (e.g., `SESS_20261101_163923_SECTIONB_ROBOTI_5AFB`).
- When a faculty member switches sections or initiates a new session, previous active sessions are gracefully concluded and archived.
- Guarantees **strictly one unique attendance record per student per session** via database compound unique indexes. Re-entry never creates duplicate attendance records.

### 2. Multi-Person Classroom Camera Processing
- Simultaneous detection and tracking of multiple individuals in the same frame using OpenCV.
- Real-time Heads-Up Display (HUD) overlay displaying session details, camera FPS, processing latency (ms), and detected participant count.
- Camera health monitoring with automatic disconnection recovery and reconnection routines.
- Operates via both physical classroom USB/IP cameras and interactive browser-based video streaming.

### 3. Privacy, Security & Informed Biometric Consent
- **Strict Consent Enforcement**: Biometric matching and identification are exclusively performed for enrolled students who have provided informed consent.
- **Identity Shielding**: Unregistered or non-consenting individuals detected in the classroom camera frame have their identities masked and protected (`Identity Protected`), with zero biometric data retained.
- **Right to Erasure**: Complete student profile correction and instant permanent deletion of facial biometric datasets via the dashboard.

### 4. Student Profile Management & Authorized Photographs
- Comprehensive student directory containing Student Name, Roll Number, Section, Department, Authorized Profile Photograph, Enrollment Status, and Consent Date.
- Authorized profile photographs displayed throughout the Live Classroom HUD, Student Notification Cards Reel, and Attendance Roster.

### 5. Session Duration Analytics & Transparent Verification
- Transparent, consent-based presence duration recording calculating exact verified presence (`first_check_in` to `last_seen`).
- Formatted duration display (`Xm Ys`) with presence status indicator.
- Single-frame transient face appearances do not constitute attendance; sustained presence or confirmed check-ins are verified.

### 6. Classroom Entry, Exit & Return Event Management
- Granular tracking of classroom movements:
  - **CHECK-IN**: Initial verified arrival in the classroom.
  - **EXIT**: Authorized temporary departure (e.g., lab, library, restroom).
  - **RETURN**: Verified return to the session (resumes presence duration without duplicating attendance).
- **Neutral Teacher Notifications**: When repeated exits (≥2) occur, the system automatically provides a neutral informational notice to the instructor for review, avoiding automated penalties.
- **Teacher Manual Correction**: Instructors can review, verify, or manually override attendance status with documented audit reasons.

### 7. Dual MongoDB & SQLite Persistence Engine
- Seamlessly connects to **MongoDB** with 5 dedicated collections:
  - `Students`: Profile information, consent status, and authorized photo paths.
  - `Classroom_Sessions`: Session metadata, schedule, faculty, and state.
  - `Attendance`: Session-scoped attendance with unique compound index `(session_id, roll_number)`.
  - `Session_Events`: Audit trail of `CHECK_IN`, `EXIT`, `RETURN`, and `MANUAL_OVERRIDE` events.
  - `System_Notifications`: Neutral instructor notices for repeated departures.
- **Zero-Configuration Fallback**: If MongoDB is not running locally, the system automatically runs on local **SQLite** (`database/students.db`) with full schema compatibility and zero data loss.
- **CSV Data Consistency**: Instant export of session attendance CSV reports matching database records.

### 8. Modern Navy-Blue & Cyan Dashboard UI
- Engineered with a high-contrast Navy-Blue (`#0a192f`, `#0d213d`) and Cyan (`#00f2fe`, `#64ffda`) theme.
- Responsive views for Live Classroom Camera, Session Management, Student Directory, Records & Duration Analytics, Enrollment, Model Training, and System Health.

---

## 🚀 Quick Localhost Startup

### Windows (One-Click Launch):
Double-click or run from command prompt:
```cmd
run.bat
```
`run.bat` activates the Python 3.10 virtual environment (`venv`), validates dependencies, tests database connectivity, starts the web server, and opens `http://127.0.0.1:5000` in your default browser.

### Cross-Platform (Windows / Linux / macOS):
```bash
python run.py
```
Or directly using the virtual environment:
```bash
# Windows
.\venv\Scripts\python app.py

# Linux / macOS
./venv/bin/python app.py
```

Application URL: **`http://127.0.0.1:5000`**

---

## 📂 Project Architecture

```text
Smart_Attendance_System/
│
├── app.py                      # Flask backend with full REST APIs & route controllers
├── run.bat                     # Windows automated launcher
├── run.py                      # Cross-platform startup script
├── classroom_camera.py         # Multi-person OpenCV camera engine & health monitor
├── database_manager.py         # Dual MongoDB & SQLite persistence manager
├── test_system.py              # Automated 10-suite unit & integration test runner
├── requirements.txt            # Compatible pinned Python packages
├── .env                        # Runtime environment configuration
├── README.md                   # System documentation
│
├── align.py                    # Dlib 68-point facial landmark alignment
├── embedding.py                # FaceNet 128-D deep embedding extractor
├── Generate_Dataset.py         # Biometric facial photo capture & enrollment
├── Model_train.py              # Neural classifier training & label serialization
├── Recognizer.py               # Legacy recognizer compatibility module
├── UI.py                       # Repaired Tkinter desktop GUI
│
├── FaceDetection/
│   ├── faces.xml               # Haar Cascade face detector
│   └── face_detection.py       # OpenCV face detection wrapper
│
├── Model_architecture/
│   └── modelArch.py            # Deep MLP Softmax classification network
│
├── MongoDB/
│   ├── retrieve_pymongo_data.py # MongoDB data retrieval utilities
│   └── Update_mongo.py          # Legacy update helpers
│
├── PreTrained_model/
│   └── facenet_keras.h5        # 92MB Inception-ResNet FaceNet model
│
├── shape_predictor_68_face_landmarks.dat # 99MB Dlib 68-point landmark weights
├── Students_Enrollment.csv     # Enrollment CSV sync registry
│
├── Model/                      # Trained classifier weights (Face_recognition.h5) & labels.json
├── people/                     # Student facial image datasets (30 frames per student)
├── database/                   # SQLite database storage (students.db) & exports/
│   └── exports/                # Session-wise attendance CSV exports
│
├── static/
│   ├── css/
│   │   └── style.css           # Navy-Blue & Cyan theme styles
│   └── profiles/               # Authorized student profile photographs
│
└── templates/                  # Bootstrap 5 Jinja2 Templates
    ├── base.html               # Master layout with navbar, active session badge, and notices
    ├── dashboard.html          # Overview dashboard with active session banner & pipeline status
    ├── classroom.html          # Live multi-person classroom camera scanner & student cards reel
    ├── sessions.html           # Classroom session lifecycle & section switcher
    ├── students.html           # Student profiles directory, photos, and consent management
    ├── records.html            # Session attendance roster, duration analytics, & exit events
    ├── enroll.html             # Interactive browser webcam student enrollment
    ├── train.html              # Asynchronous neural classifier training interface
    ├── database_status.html    # Database connection & schema inspector
    └── settings.html           # Camera index, thresholds, and operational parameters
```

---

## 🧪 Comprehensive Verification Suite

Run all 10 automated test suites:
```bash
.\venv\Scripts\python -m unittest test_system.py
```

### Verified Test Capabilities:
1. `test_01_required_files_and_models`: Verifies FaceNet, Dlib landmarks, Haar cascade, and critical modules.
2. `test_02_session_creation_and_section_switch`: Validates session creation, section switching, unique `session_id`, and active status transitions.
3. `test_03_student_profile_management_and_consent`: Tests profile enrollment, editing, and consent privacy enforcement.
4. `test_04_session_attendance_and_duplicate_prevention`: Proves single attendance record per session and duplicate prevention on re-detection.
5. `test_05_checkin_exit_and_return_lifecycle`: Tests exit tracking, return event logging, and status transitions without duplicate record creation.
6. `test_06_neutral_notification_on_repeated_exits`: Validates neutral teacher notices generated on repeated exits and read status updating.
7. `test_07_session_duration_analytics`: Verifies transparent presence duration calculation and formatting.
8. `test_08_multiple_person_camera_processing`: Validates OpenCV multi-person face processing, HUD metrics, and camera health monitoring.
9. `test_09_manual_teacher_correction`: Verifies instructor manual status overrides and audit event logging.
10. `test_10_csv_exports_and_flask_routes`: Tests session CSV export generation and verifies that all 18 web endpoints return HTTP 200.

---

## 🔒 Privacy & Compliance Architecture

- **Participant Consent**: Attendance and facial identification are restricted to consenting individuals. Biometric data is stored locally and securely.
- **Passive Privacy Masking**: Non-consenting or unregistered faces are identified as `Identity Protected` and masked in real-time camera views.
- **Audit Trails**: All status overrides, check-ins, exits, and returns are permanently recorded in `session_events` with timestamps and instructor verification notes.
- **Right to Erasure**: Biometric images in `people/<roll>_<name>` and profile metadata can be erased immediately via `/api/students/delete`.
