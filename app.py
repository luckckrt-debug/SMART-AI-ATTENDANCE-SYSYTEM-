import os
import sys
import json
import base64
import threading
import datetime
from pathlib import Path
from functools import wraps
import cv2
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
    Response,
    redirect,
    url_for,
    flash,
    session,
    abort
)
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from database_manager import db_manager, PROFILES_DIR
from Generate_Dataset import save_student_captured_images, Generate_Data
from Model_train import Model_Training, TRAINING_STATE, parse_student_folder
from classroom_camera import classroom_camera
from Recognizer import FaceRecognizer

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "smart-classroom-super-secret-key-2026")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.permanent_session_lifetime = datetime.timedelta(days=7)

# Register AI Training Studio & Visual Analytics Blueprint
from studio_routes import studio_bp
app.register_blueprint(studio_bp)


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

CONFIG = {
    "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.85")),
    "frames_for_attendance": int(os.getenv("FRAMES_FOR_ATTENDANCE", "6")),
    "camera_index": int(os.getenv("CAMERA_INDEX", "0"))
}

# Update camera engine parameters
classroom_camera.confidence_threshold = CONFIG["confidence_threshold"]
classroom_camera.consecutive_frames_needed = CONFIG["frames_for_attendance"]

# Global legacy recognizer instance
legacy_recognizer = FaceRecognizer(
    confidence_threshold=CONFIG["confidence_threshold"],
    consecutive_frames_needed=CONFIG["frames_for_attendance"]
)


# -------------------------------------------------------------
# ROLE-BASED ACCESS CONTROL (RBAC) HELPERS
# -------------------------------------------------------------
def get_current_user():
    """Returns active user session, defaulting to Teacher for classroom access."""
    if "user" not in session:
        session["user"] = {
            "username": "teacher",
            "role": "TEACHER",
            "name": "Prof. Sharma",
            "email": "sharma@smartclassroom.edu"
        }
    return session["user"]


@app.context_processor
def inject_user():
    user = get_current_user()
    return {
        "current_user": user,
        "is_admin": user.get("role") == "ADMIN"
    }


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login_page", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Protects administrator endpoints and pages. Returns 403 Forbidden to teachers."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user.get("role") != "ADMIN":
            if request.path.startswith("/api/"):
                return jsonify({
                    "success": False,
                    "error": "FORBIDDEN",
                    "message": "Access Denied: Administrator role required for this action."
                }), 403
            flash("Access Restricted: Administrator privileges required to access the Admin Panel.", "warning")
            return render_template("403.html", user=user), 403
        return f(*args, **kwargs)
    return decorated_function


# -------------------------------------------------------------
# System Health Helper
# -------------------------------------------------------------
def get_system_health():
    db_status = db_manager.get_status()
    facenet_exists = (BASE_DIR / "PreTrained_model" / "facenet_keras.h5").exists()
    landmarks_exists = (BASE_DIR / "shape_predictor_68_face_landmarks.dat").exists()
    cascade_exists = (BASE_DIR / "FaceDetection" / "faces.xml").exists()
    classifier_exists = (BASE_DIR / "Model" / "Face_recognition.h5").exists() or \
                        (BASE_DIR / "Model" / "Face_recognition.MODEL").exists()
    labels_exists = (BASE_DIR / "Model" / "labels.json").exists()

    students = db_manager.get_all_students()
    active_sess = db_manager.get_active_session()
    sess_id = active_sess["session_id"] if active_sess else None
    roster = db_manager.get_session_roster_status(sess_id) if sess_id else []

    return {
        "db": db_status,
        "models": {
            "facenet": facenet_exists,
            "landmarks": landmarks_exists,
            "cascade": cascade_exists,
            "classifier": classifier_exists and labels_exists,
        },
        "stats": {
            "total_students": len(students),
            "active_present": len([s for s in roster if s["is_present"]]),
            "active_absent": len([s for s in roster if not s["is_present"]]),
            "recent_logs": db_manager.get_recent_logs(10)
        }
    }


# -------------------------------------------------------------
# AUTHENTICATION & ROLE SWITCHING ROUTES
# -------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        user = db_manager.authenticate_user(u, p)
        if user:
            session.permanent = True
            session["user"] = user
            flash(f"Welcome back, {user['name']} ({user['role']})!", "success")
            if user["role"] == "ADMIN":
                return redirect(url_for("admin_page"))
            return redirect(url_for("index"))
        flash("Invalid username or password. Default accounts: 'teacher' / 'teacher123' or 'admin' / 'admin123'.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout_page():
    session.pop("user", None)
    flash("Signed out successfully.", "info")
    return redirect(url_for("login_page"))


@app.route("/api/auth/switch_role/<role>")
def api_switch_role(role):
    r = role.upper()
    if r == "ADMIN":
        session.permanent = True
        session["user"] = {
            "username": "admin",
            "role": "ADMIN",
            "name": "Administrator",
            "email": "admin@smartclassroom.edu"
        }
        flash("Switched to Administrator role (Full access enabled).", "success")
        return redirect(url_for("admin_page"))
    else:
        session["user"] = {
            "username": "teacher",
            "role": "TEACHER",
            "name": "Prof. Sharma",
            "email": "sharma@smartclassroom.edu"
        }
        flash("Switched to Teacher role (Classroom Attendance & Session Manager view).", "info")
        return redirect(url_for("index"))


# -------------------------------------------------------------
# STATIC PHOTO SERVING FOR PEOPLE FOLDER
# -------------------------------------------------------------
@app.route("/people_photos/<path:filepath>")
def serve_people_photo(filepath):
    return send_from_directory(BASE_DIR / "people", filepath)


# -------------------------------------------------------------
# A. HOME SCREEN — TEACHER DASHBOARD
# Contains ONLY: 1. ATTENDANCE & 2. SESSION MANAGER
# -------------------------------------------------------------
@app.route("/")
@app.route("/dashboard")
def index():
    all_sessions = db_manager.get_all_sessions(limit=30)
    active_sess = db_manager.get_active_session()

    selected_sess_id = request.args.get("session_id")
    if not selected_sess_id:
        sess_id = active_sess["session_id"] if active_sess else (all_sessions[0]["session_id"] if all_sessions else None)
    else:
        sess_id = selected_sess_id

    selected_session = db_manager.get_session(sess_id) if sess_id else active_sess
    roster = db_manager.get_session_roster_status(sess_id) if sess_id else []
    events = db_manager.get_session_events(sess_id, limit=30) if sess_id else []

    classes = db_manager.get_all_classes()
    subjects = db_manager.get_all_subjects()
    teachers = db_manager.get_all_teachers()

    stats = {
        "total_students": len(roster),
        "active_present": len([s for s in roster if s["is_present"]]),
        "active_absent": len([s for s in roster if not s["is_present"]]),
    }

    return render_template(
        "dashboard.html",
        active_session=active_sess,
        selected_session=selected_session,
        selected_session_id=sess_id,
        roster=roster,
        events=events,
        classes=classes,
        subjects=subjects,
        teachers=teachers,
        sessions=all_sessions,
        stats=stats,
        config=CONFIG
    )


# -------------------------------------------------------------
# B. SEPARATE ADMIN PANEL (PROTECTED)
# Contains all 10 administrative & configuration modules
# -------------------------------------------------------------
@app.route("/admin")
@admin_required
def admin_page():
    health = get_system_health()
    students = db_manager.get_all_students()
    classes = db_manager.get_all_classes()
    subjects = db_manager.get_all_subjects()
    teachers = db_manager.get_all_teachers()
    users = db_manager.get_all_users()
    all_sessions = db_manager.get_all_sessions(limit=50)
    cam_health = classroom_camera.get_health()
    faculty = db_manager.get_all_faculty()
    class_groups = db_manager.get_all_class_groups()
    recognition_logs = db_manager.get_recognition_logs(limit=50)
    analytics = db_manager.get_dashboard_analytics()

    return render_template(
        "admin.html",
        health=health,
        students=students,
        classes=classes,
        subjects=subjects,
        teachers=teachers,
        faculty=faculty,
        users=users,
        all_sessions=all_sessions,
        camera_health=cam_health,
        training_state=TRAINING_STATE,
        class_groups=class_groups,
        recognition_logs=recognition_logs,
        analytics=analytics,
        config=CONFIG
    )


# -------------------------------------------------------------
# LIVE CLASSROOM CAMERA & RECORDS PAGES
# -------------------------------------------------------------
@app.route("/classroom")
def classroom_page():
    active_sess = db_manager.get_active_session()
    if not active_sess:
        active_id = db_manager.create_session(
            academic_year="2026-2027",
            department="Computer Science & Engineering",
            class_name="B.Tech Year 3",
            section="Section A",
            subject="Machine Learning & AI",
            faculty_name="Prof. Sharma"
        )
        active_sess = db_manager.get_active_session()

    sess_id = active_sess["session_id"]
    classroom_camera.active_session_id = sess_id
    roster = db_manager.get_session_roster_status(sess_id, section=active_sess.get("section"))
    events = db_manager.get_session_events(sess_id, limit=20)
    notifications = db_manager.get_session_notifications(sess_id, unread_only=True)
    health = get_system_health()

    return render_template(
        "classroom.html",
        session=active_sess,
        active_session=active_sess,
        roster=roster,
        events=events,
        notifications=notifications,
        health=health,
        config=CONFIG
    )


@app.route("/records")
def records_page():
    all_sessions = db_manager.get_all_sessions(50)
    active_sess = db_manager.get_active_session()

    selected_session_id = request.args.get("session_id")
    if not selected_session_id:
        if active_sess:
            selected_session_id = active_sess["session_id"]
        elif all_sessions:
            selected_session_id = all_sessions[0]["session_id"]
        else:
            selected_session_id = None

    selected_session = db_manager.get_session(selected_session_id) if selected_session_id else None
    roster = db_manager.get_session_roster_status(selected_session_id) if selected_session_id else []
    events = db_manager.get_session_events(selected_session_id, limit=50) if selected_session_id else []
    health = get_system_health()

    hindi_records = db_manager.get_attendance("Hindi")
    english_records = db_manager.get_attendance("English")

    return render_template(
        "records.html",
        active_session=active_sess,
        selected_session_id=selected_session_id,
        selected_session=selected_session,
        all_sessions=all_sessions,
        roster=roster,
        events=events,
        hindi_records=hindi_records,
        english_records=english_records,
        logs=db_manager.get_recent_logs(50),
        health=health
    )


# -------------------------------------------------------------
# LEGACY COMPATIBILITY REDIRECTS
# -------------------------------------------------------------
@app.route("/database-status")
def legacy_database_status():
    return redirect("/admin#pane-database")


@app.route("/settings")
def legacy_settings():
    return redirect("/admin#pane-database")


@app.route("/sessions")
def legacy_sessions():
    return redirect(url_for("index"))


@app.route("/students")
@app.route("/enroll")
def legacy_students():
    return redirect("/admin#pane-students")


@app.route("/train")
def legacy_train():
    return redirect("/admin#pane-ai")


@app.route("/attendance/<subject>")
def legacy_subject_attendance(subject):
    return redirect(url_for("records_page"))


# -------------------------------------------------------------
# ADMINISTRATIVE APIs (Protected by @admin_required)
# -------------------------------------------------------------
@app.route("/api/admin/classes/create", methods=["POST"])
@admin_required
def api_admin_class_create():
    data = request.get_json(force=True)
    c_name = data.get("class_name")
    dept = data.get("department")
    sec = data.get("section")
    if not c_name or not dept or not sec:
        return jsonify({"success": False, "message": "All fields required."}), 400
    ok, msg = db_manager.create_class(c_name, dept, sec)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/classes/<class_id>/delete", methods=["POST"])
@admin_required
def api_admin_class_delete(class_id):
    ok, msg = db_manager.delete_class(class_id)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/subjects/create", methods=["POST"])
@admin_required
def api_admin_subject_create():
    data = request.get_json(force=True)
    s_name = data.get("subject_name")
    s_code = data.get("subject_code")
    dept = data.get("department")
    if not s_name or not dept:
        return jsonify({"success": False, "message": "Subject name and department required."}), 400
    ok, msg = db_manager.create_subject(s_name, dept, s_code)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/subjects/<subject_id>/delete", methods=["POST"])
@admin_required
def api_admin_subject_delete(subject_id):
    ok, msg = db_manager.delete_subject(subject_id)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/teachers/create", methods=["POST"])
@admin_required
def api_admin_teacher_create():
    data = request.get_json(force=True)
    name = data.get("name")
    dept = data.get("department")
    email = data.get("email")
    phone = data.get("phone")
    if not name or not dept:
        return jsonify({"success": False, "message": "Teacher name and department required."}), 400
    ok, msg = db_manager.create_teacher(name, dept, email, phone)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/teachers/<teacher_id>/delete", methods=["POST"])
@admin_required
def api_admin_teacher_delete(teacher_id):
    ok, msg = db_manager.delete_teacher(teacher_id)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/users/create", methods=["POST"])
@admin_required
def api_admin_user_create():
    data = request.get_json(force=True)
    u = data.get("username")
    p = data.get("password")
    r = data.get("role", "TEACHER")
    n = data.get("name")
    em = data.get("email")
    ok, msg = db_manager.create_user(u, p, r, n, em)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/users/<username>/delete", methods=["POST"])
@admin_required
def api_admin_user_delete(username):
    ok, msg = db_manager.delete_user(username)
    return jsonify({"success": ok, "message": msg})


# Student Photo Intake (Up to 10 photos, validation, deletion, replacement)
@app.route("/api/admin/students/<roll>/photos", methods=["GET"])
@admin_required
def api_admin_student_photos(roll):
    photos = db_manager.get_student_photos(roll)
    return jsonify({"success": True, "photos": photos})


@app.route("/api/admin/students/<roll>/photos/upload", methods=["POST"])
@admin_required
def api_admin_student_photo_upload(roll):
    # Security Rule: Only registered candidates are eligible for biometric photo upload
    student = db_manager.get_student(roll)
    if not student:
        return jsonify({
            "success": False,
            "message": f"Biometric Security Policy: Candidate with Roll ID '{roll}' is not registered. Photos can strictly be uploaded ONLY for registered candidates. Please enroll the candidate under Student Management first."
        }), 400

    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No photo file provided."}), 400
    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty file."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return jsonify({"success": False, "message": "Invalid format. Supported formats: JPEG, PNG."}), 400

    file_bytes = file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return jsonify({"success": False, "message": "File exceeds 10MB limit."}), 400

    ok, msg = db_manager.add_student_photo(roll, file_bytes, file.filename)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/students/<roll>/photos/<filename>/delete", methods=["POST"])
@admin_required
def api_admin_student_photo_delete(roll, filename):
    ok, msg = db_manager.delete_student_photo(roll, filename)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/students/<roll>/photos/<filename>/replace", methods=["POST"])
@admin_required
def api_admin_student_photo_replace(roll, filename):
    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No replacement photo provided."}), 400
    file = request.files["photo"]
    file_bytes = file.read()
    ok, msg = db_manager.replace_student_photo(roll, filename, file_bytes)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/camera/config", methods=["POST"])
@admin_required
def api_admin_camera_config():
    global CONFIG
    data = request.get_json(force=True)
    idx = data.get("camera_index", CONFIG["camera_index"])
    conf = data.get("confidence_threshold", CONFIG["confidence_threshold"])
    frames = data.get("frames_for_attendance", CONFIG["frames_for_attendance"])

    CONFIG["camera_index"] = int(idx)
    CONFIG["confidence_threshold"] = float(conf)
    CONFIG["frames_for_attendance"] = int(frames)

    classroom_camera.camera_index = CONFIG["camera_index"]
    classroom_camera.confidence_threshold = CONFIG["confidence_threshold"]
    classroom_camera.consecutive_frames_needed = CONFIG["frames_for_attendance"]

    return jsonify({"success": True, "message": "Camera configuration saved."})


# -------------------------------------------------------------
# CLASSROOM SESSION APIS (Teacher & Admin)
# -------------------------------------------------------------
@app.route("/api/sessions/create", methods=["POST"])
def api_sessions_create():
    try:
        data = request.get_json(force=True)
        deg = data.get("degree") or ("B.Tech" if "b.tech" in str(data.get("class_name", "")).lower() else "B.Tech")
        strm = data.get("stream") or data.get("department", "Computer Science & Engineering")
        ayear = data.get("academic_year") or data.get("class_name", "2nd Year")
        asess = data.get("academic_session") or "2026-27"
        subj = data.get("subject", "Machine Learning & AI")
        faculty = data.get("faculty_name", "Prof. Sharma")
        fid = data.get("faculty_id", "")
        sec = data.get("section", "")

        new_sess_id = db_manager.create_session(
            degree=deg,
            stream=strm,
            academic_year=ayear,
            academic_session=asess,
            department=strm,
            class_name=f"{deg} {ayear}",
            section=sec,
            subject=subj,
            faculty_name=faculty,
            faculty_id=fid
        )
        classroom_camera.active_session_id = new_sess_id

        return jsonify({
            "success": True,
            "session_id": new_sess_id,
            "message": f"Classroom session {new_sess_id} started."
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sessions/end", methods=["POST"])
def api_sessions_end():
    db_manager.end_active_sessions()
    classroom_camera.active_session_id = None
    return jsonify({"success": True, "message": "Classroom session concluded."})


@app.route("/api/classroom/frame", methods=["POST"])
def api_classroom_frame():
    try:
        data = request.get_json(force=True)
        frame_b64 = data.get("frame", "")
        session_id = data.get("session_id")

        if not frame_b64:
            return jsonify({"success": False, "message": "No frame data"}), 400

        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        frame_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "message": "Could not decode frame"}), 400

        if not session_id:
            active_s = db_manager.get_active_session()
            session_id = active_s["session_id"] if active_s else None

        annotated, detections = classroom_camera.process_frame(
            frame,
            auto_mark=True,
            active_session_id=session_id
        )

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        # Compute real-time HUD counters
        faculty_count = sum(1 for d in detections if d.get("is_faculty"))
        outside_count = sum(1 for d in detections if d.get("is_outside_class"))
        unknown_count = sum(1 for d in detections if d.get("is_unknown") or d.get("recognition_state") == "UNKNOWN")
        verified_count = sum(1 for d in detections if d.get("verified"))

        eligible_ids = db_manager.get_eligible_students_for_session(session_id) if session_id else []
        present_count = 0
        if session_id:
            try:
                att_records = db_manager.get_session_attendance(session_id)
                present_count = len(att_records)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "annotated_frame": f"data:image/jpeg;base64,{annotated_b64}",
            "detections": detections,
            "faces_count": len(detections),
            "session_id": session_id,
            "status": "ONLINE",
            "latency_ms": round(classroom_camera.latency_ms, 1),
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "hud_counts": {
                "eligible": len(eligible_ids),
                "present": present_count,
                "verified_in_frame": verified_count,
                "faculty_excluded": faculty_count,
                "outside_class": outside_count,
                "unknown": unknown_count
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/exit", methods=["POST"])
def api_attendance_exit():
    try:
        data = request.get_json(force=True)
        sess_id = data.get("session_id")
        roll = data.get("roll_number")
        reason = data.get("reason", "Classroom exit")
        success, msg = db_manager.record_student_exit(sess_id, roll, reason=reason)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/return", methods=["POST"])
def api_attendance_return():
    try:
        data = request.get_json(force=True)
        sess_id = data.get("session_id")
        roll = data.get("roll_number")
        success, msg, action = db_manager.record_session_presence(sess_id, roll, verification_method="MANUAL_CONFIRMATION")
        return jsonify({"success": success, "message": msg, "action": action})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/attendance/correct", methods=["POST"])
def api_attendance_correct():
    try:
        data = request.get_json(force=True)
        sess_id = data.get("session_id")
        roll = data.get("roll_number")
        status = data.get("status", "PRESENT")
        reason = data.get("reason", "Instructor verification")
        success, msg = db_manager.manual_teacher_correction(sess_id, roll, status, reason=reason)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/sessions/<session_id>/notifications")
def api_session_notifications(session_id):
    notifs = db_manager.get_session_notifications(session_id)
    return jsonify({"success": True, "notifications": notifs})


@app.route("/api/notifications/<notification_id>/read", methods=["POST"])
def api_notification_read(notification_id):
    db_manager.mark_notification_read(notification_id)
    return jsonify({"success": True})


@app.route("/export/session_csv/<session_id>")
def export_session_csv_route(session_id):
    csv_file = db_manager.export_session_csv(session_id)
    if os.path.exists(csv_file):
        return send_file(csv_file, as_attachment=True, download_name=f"Session_{session_id}_Attendance.csv", mimetype="text/csv")
    return "Export failed", 404


# -------------------------------------------------------------
# PROTECTED DATA-ENTRY & CONFIGURATION APIS (@admin_required)
# -------------------------------------------------------------
@app.route("/api/enroll", methods=["POST"])
@app.route("/api/admin/students/register", methods=["POST"])
@admin_required
def api_enroll():
    try:
        if request.is_json:
            data = request.get_json(force=True)
            files = []
        else:
            data = request.form.to_dict()
            files = request.files.getlist("photos") or request.files.getlist("photo") or request.files.getlist("images")

        name = data.get("name", "").strip()
        roll = data.get("roll_number", "").strip() or data.get("student_id", "").strip()
        email = data.get("email", "").strip() or f"{roll.lower()}@college.edu"
        mobile = data.get("mobile", "").strip()
        degree = data.get("degree", "").strip() or "B.Tech"
        stream = data.get("stream", "").strip() or data.get("department", "Computer Science & Engineering").strip()
        academic_year = data.get("academic_year", "").strip() or "2nd Year"
        academic_session = data.get("academic_session", "").strip() or "2026-27"
        consent = bool(data.get("consent_given", True))
        images = data.get("images", []) or data.get("photos", [])

        if not name or not roll:
            return jsonify({"success": False, "message": "Student name and roll number / Student ID are required."}), 400

        if not degree or not stream or not academic_year or not academic_session:
            return jsonify({
                "success": False,
                "message": "Degree, Stream/Branch, Academic Year, and Academic Session are mandatory."
            }), 400

        prof_photo_path = f"/static/profiles/{roll}.jpg"
        success, msg = db_manager.enroll_student(
            roll_number=roll,
            name=name,
            email=email,
            mobile=mobile,
            degree=degree,
            stream=stream,
            academic_year=academic_year,
            academic_session=academic_session,
            department=stream,
            section="",
            profile_photo=prof_photo_path,
            sync_csv=True,
            consent_given=consent,
            face_enrollment_status="PENDING"
        )
        if not success:
            return jsonify({"success": False, "message": msg}), 400

        # Automatic face enrollment if photos provided
        enroll_stages = [
            {"stage": "Uploading", "status": "completed"},
            {"stage": "Processing Image", "status": "pending"},
            {"stage": "Detecting Face", "status": "pending"},
            {"stage": "Generating Face Embedding", "status": "pending"},
            {"stage": "Validating Face Data", "status": "pending"},
            {"stage": "Face Enrollment Successful", "status": "pending"}
        ]

        from ai_engine import auto_enroll_student_face
        face_status = "PENDING"
        enrollment_result = None

        all_photo_bytes = []
        for f in files:
            b = f.read()
            if b:
                all_photo_bytes.append((b, f.filename))

        if isinstance(images, list):
            for idx, img_item in enumerate(images):
                if isinstance(img_item, str) and img_item.strip():
                    img_b64 = img_item
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    try:
                        b = base64.b64decode(img_b64)
                        all_photo_bytes.append((b, f"photo_{idx+1}.jpg"))
                    except Exception:
                        pass

        if all_photo_bytes:
            enroll_stages[1]["status"] = "completed"
            for b, fname in all_photo_bytes:
                db_manager.add_student_photo(roll, b, fname)

            first_bytes, _ = all_photo_bytes[0]
            enroll_stages[2]["status"] = "completed"
            enroll_stages[3]["status"] = "completed"
            res = auto_enroll_student_face(roll, name, first_bytes, source="registration")
            enrollment_result = res
            if res.get("success"):
                enroll_stages[4]["status"] = "completed"
                enroll_stages[5]["status"] = "completed"
                face_status = "Face Enrollment Successful"
            else:
                face_status = res.get("message", "Face enrollment problem")
        else:
            db_manager._get_student_folder(roll, create=True)

        student_rec = db_manager.get_student(roll)
        return jsonify({
            "success": True,
            "message": f"Student {name} (ID: {roll}) registered successfully in {degree} {stream} ({academic_year} {academic_session}).",
            "student": student_rec,
            "stages": enroll_stages,
            "face_enrollment_status": face_status,
            "enrollment_result": enrollment_result
        })
    except Exception as e:
        return jsonify({"success": False, "message": f"Server enrollment error: {str(e)}"}), 500


# -------------------------------------------------------------
# FACULTY MANAGEMENT & EXCLUSION APIS (@admin_required)
# -------------------------------------------------------------
@app.route("/api/admin/faculty", methods=["GET"])
@admin_required
def api_admin_get_faculty():
    faculty = db_manager.get_all_faculty()
    return jsonify({"success": True, "faculty": faculty})


@app.route("/api/admin/faculty/create", methods=["POST"])
@app.route("/api/admin/faculty/register", methods=["POST"])
@admin_required
def api_admin_faculty_create():
    try:
        if request.is_json:
            data = request.get_json(force=True)
            files = []
        else:
            data = request.form.to_dict()
            files = request.files.getlist("photos") or request.files.getlist("photo")

        name = data.get("name", "").strip()
        faculty_id = data.get("faculty_id", "").strip() or data.get("id", "").strip()
        email = data.get("email", "").strip()
        dept = data.get("department", "Computer Science & Engineering").strip()
        desig = data.get("designation", "Faculty Member").strip()
        phone = data.get("phone", "").strip()
        images = data.get("images", []) or data.get("photos", [])

        if not name or not faculty_id:
            return jsonify({"success": False, "message": "Faculty name and Faculty ID are required."}), 400

        ok, msg = db_manager.enroll_faculty(
            faculty_id=faculty_id,
            name=name,
            email=email,
            department=dept,
            designation=desig,
            phone=phone
        )
        if not ok:
            return jsonify({"success": False, "message": msg}), 400

        for f in files:
            b = f.read()
            if b:
                db_manager.add_faculty_photo(faculty_id, b, f.filename)

        if isinstance(images, list):
            for idx, img_item in enumerate(images):
                if isinstance(img_item, str) and img_item.strip():
                    img_b64 = img_item
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    try:
                        b = base64.b64decode(img_b64)
                        db_manager.add_faculty_photo(faculty_id, b, f"faculty_{idx+1}.jpg")
                    except Exception:
                        pass

        faculty_rec = db_manager.get_faculty(faculty_id)
        return jsonify({
            "success": True,
            "message": f"Faculty {name} ({faculty_id}) registered successfully in isolated exclusion namespace.",
            "faculty": faculty_rec
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/faculty/<faculty_id>", methods=["GET"])
@admin_required
def api_admin_get_faculty_single(faculty_id):
    faculty = db_manager.get_faculty(faculty_id)
    if not faculty:
        return jsonify({"success": False, "message": "Faculty member not found."}), 404
    return jsonify({"success": True, "faculty": faculty})


@app.route("/api/admin/faculty/<faculty_id>/delete", methods=["POST", "DELETE"])
@admin_required
def api_admin_delete_faculty(faculty_id):
    ok, msg = db_manager.delete_faculty(faculty_id)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/faculty/<faculty_id>/photos", methods=["GET"])
@admin_required
def api_admin_get_faculty_photos(faculty_id):
    photos = db_manager.get_faculty_photos(faculty_id)
    return jsonify({"success": True, "photos": photos})


@app.route("/api/admin/faculty/<faculty_id>/photos/upload", methods=["POST"])
@admin_required
def api_admin_upload_faculty_photo(faculty_id):
    faculty = db_manager.get_faculty(faculty_id)
    if not faculty:
        return jsonify({"success": False, "message": f"Faculty member with ID '{faculty_id}' not found."}), 404

    if "photo" not in request.files:
        return jsonify({"success": False, "message": "No photo file provided."}), 400
    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"success": False, "message": "Empty file."}), 400

    file_bytes = file.read()
    ok, msg = db_manager.add_faculty_photo(faculty_id, file_bytes, file.filename)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/admin/faculty/<faculty_id>/photos/<filename>/delete", methods=["POST", "DELETE"])
@admin_required
def api_admin_delete_faculty_photo(faculty_id, filename):
    ok, msg = db_manager.delete_faculty_photo(faculty_id, filename)
    return jsonify({"success": ok, "message": msg})


# -------------------------------------------------------------
# CLASS GROUPS, RECOGNITION LOGS & ANALYTICS APIS (@admin_required)
# -------------------------------------------------------------
@app.route("/api/admin/classes/groups", methods=["GET"])
@admin_required
def api_admin_class_groups():
    groups = db_manager.get_all_class_groups()
    return jsonify({"success": True, "class_groups": groups})


@app.route("/api/admin/recognition-logs", methods=["GET"])
@admin_required
def api_admin_recognition_logs():
    limit = int(request.args.get("limit", 100))
    logs = db_manager.get_recognition_logs(limit=limit)
    return jsonify({"success": True, "logs": logs})


@app.route("/api/admin/analytics", methods=["GET"])
@admin_required
def api_admin_analytics():
    analytics = db_manager.get_dashboard_analytics()
    return jsonify({"success": True, "analytics": analytics})


@app.route("/api/student/<roll_number>", methods=["DELETE"])
@admin_required
def api_delete_student(roll_number):
    try:
        success, msg = db_manager.delete_student(roll_number)
        return jsonify({"success": success, "message": msg})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/train/start", methods=["POST"])
@admin_required
def api_train_start():
    if TRAINING_STATE["is_training"]:
        return jsonify({"success": False, "message": "Training is already in progress."}), 400

    def run_training_async():
        success, msg = Model_Training(epochs=60)
        classroom_camera.load_models()
        legacy_recognizer._load_recognition_model()

    thread = threading.Thread(target=run_training_async, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "AI model training started in background."})


@app.route("/api/train/status")
def api_train_status():
    return jsonify(TRAINING_STATE)


@app.route("/api/db/sync_csv", methods=["POST"])
@admin_required
def api_sync_csv():
    try:
        db_manager.sync_from_csv()
        return jsonify({"success": True, "message": "Synchronized database with Students_Enrollment.csv."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/db/test_mongo", methods=["POST"])
@admin_required
def api_test_mongo():
    try:
        data = request.get_json(force=True, silent=True) or {}
        uri = data.get("mongo_uri")
        db_name = data.get("mongo_db")
        success, message = db_manager.test_or_reconnect_mongo(uri, db_name)
        status = db_manager.get_status()
        return jsonify({"success": success, "message": message, "status": status})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/export/csv/<subject>")
def export_csv_route(subject):
    subj = subject.strip().capitalize()
    if subj in ["Hindi", "English"]:
        csv_path = db_manager.export_csv(subj)
        filename = f"{subj}_Attendance.csv"
    elif subj == "Enrollment":
        csv_path = str(BASE_DIR / "Students_Enrollment.csv")
        filename = "Students_Enrollment.csv"
    else:
        return "Invalid subject requested", 400

    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name=filename, mimetype="text/csv")
    return "File not found", 404


@app.route("/api/status")
def api_status():
    return jsonify(get_system_health())



# -------------------------------------------------------------
# ADVANCED AI MODEL TRAINING & DATASET MANAGEMENT ENDPOINTS
# -------------------------------------------------------------
@app.route("/api/db/reconnect", methods=["POST"])
@admin_required
def api_reconnect_mongo():
    try:
        success, message = db_manager.test_or_reconnect_mongo()
        status = db_manager.get_status()
        return jsonify({"success": success, "message": message, "status": status})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/overview", methods=["GET"])
@admin_required
def api_admin_model_overview():
    try:
        from training_pipeline import model_version_manager
        from ai_engine import prototype_matcher
        active = model_version_manager.get_active_version()
        status = db_manager.get_status()
        all_students = db_manager.get_all_students()

        return jsonify({
            "success": True,
            "active_version": active.get("version_id", "v1.0-default"),
            "architecture": "FaceNet-128D Inception-ResNet + Hyperspherical Prototype Matcher + YuNet FaceDetectorYN",
            "created_at": active.get("created_at", "N/A"),
            "total_identities": len(prototype_matcher.prototypes),
            "total_registered_students": len(all_students),
            "threshold_high": prototype_matcher.threshold_high,
            "threshold_review": prototype_matcher.threshold_review,
            "val_accuracy": active.get("val_accuracy", 0.0),
            "status": active.get("status", "ACTIVE"),
            "db_mode": status.get("mode", "MongoDB"),
            "is_mongo_connected": status.get("is_mongo_connected", False)
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/dataset/audit", methods=["GET"])
@admin_required
def api_admin_dataset_audit():
    try:
        from dataset_manager import DatasetManager
        mgr = DatasetManager()
        report = mgr.audit_dataset()
        return jsonify({"success": True, "report": report})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/dataset/group-photo", methods=["POST"])
@admin_required
def api_admin_group_photo():
    try:
        from dataset_manager import DatasetManager
        if "photo" not in request.files:
            return jsonify({"success": False, "message": "No file uploaded"}), 400
        f = request.files["photo"]
        img_bytes = f.read()
        mgr = DatasetManager()
        result = mgr.extract_faces_from_group_photo(img_bytes)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/dataset/assign-crop", methods=["POST"])
@admin_required
def api_admin_assign_crop():
    try:
        from dataset_manager import DatasetManager
        data = request.get_json(force=True, silent=True) or {}
        roll = data.get("roll_number")
        crop_b64 = data.get("crop_b64")
        if not roll or not crop_b64:
            return jsonify({"success": False, "message": "roll_number and crop_b64 required"}), 400
        mgr = DatasetManager()
        res = mgr.assign_crop_to_student(roll, crop_b64)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/dataset/delete-image", methods=["POST"])
@admin_required
def api_admin_delete_image():
    try:
        from dataset_manager import DatasetManager
        data = request.get_json(force=True, silent=True) or {}
        roll = data.get("roll_number")
        fname = data.get("filename")
        if not roll or not fname:
            return jsonify({"success": False, "message": "roll_number and filename required"}), 400
        mgr = DatasetManager()
        res = mgr.delete_image(roll, fname)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/train", methods=["POST"])
@admin_required
def api_admin_model_train():
    try:
        from training_pipeline import start_training_job_async
        job_id = start_training_job_async()
        return jsonify({"success": True, "job_id": job_id, "message": "AI Model Training pipeline launched in background."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/train/status", methods=["GET"])
@admin_required
def api_admin_train_status():
    try:
        from training_pipeline import get_training_status
        status = get_training_status()
        return jsonify({"success": True, "status": status})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/versions", methods=["GET"])
@admin_required
def api_admin_model_versions():
    try:
        from training_pipeline import model_version_manager
        history = model_version_manager.get_history()
        return jsonify({"success": True, "versions": history})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/versions/<version_id>/deploy", methods=["POST"])
@admin_required
def api_admin_deploy_version(version_id):
    try:
        from training_pipeline import model_version_manager
        res = model_version_manager.deploy_version(version_id)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/versions/<version_id>/rollback", methods=["POST"])
@admin_required
def api_admin_rollback_version(version_id):
    try:
        from training_pipeline import model_version_manager
        res = model_version_manager.rollback(version_id)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/admin/model/test-photo", methods=["POST"])
@admin_required
def api_admin_test_photo():
    try:
        from ai_engine import face_detector, align_face_crop, assess_face_quality, prototype_matcher
        from embedding import emb
        import cv2
        import numpy as np
        import base64

        if "photo" not in request.files:
            return jsonify({"success": False, "message": "No test image file uploaded."}), 400

        f = request.files["photo"]
        img_bytes = f.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "message": "Invalid or unreadable image file."}), 400

        faces = face_detector.detect(img)
        e_model = emb()
        detections = []
        annotated = img.copy()

        for idx, f_info in enumerate(faces):
            box = f_info["bbox"]
            lmks = f_info["landmarks"]
            crop = align_face_crop(img, box, lmks, output_size=(160, 160))
            quality = assess_face_quality(crop if crop is not None else img, box)

            identity = "UNKNOWN"
            name = "Unregistered Person"
            confidence = 0.0
            state = "UNKNOWN"

            if crop is not None:
                prep = e_model.preprocess_face(crop)
                if prep is not None:
                    vec = e_model.calculate(prep)
                    identity, name, confidence, state = prototype_matcher.match_embedding(vec)

            x, y, w, h = box
            if state == "HIGH_CONFIDENCE":
                box_color = (46, 204, 113)   # Green
            elif state == "REVIEW_REQUIRED":
                box_color = (0, 165, 255)    # Orange
            else:
                box_color = (0, 0, 230)      # Red

            cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
            label = f"{name} ({confidence*100:.1f}%) [{state}]"
            cv2.putText(annotated, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            detections.append({
                "face_index": idx + 1,
                "bbox": box,
                "roll_number": identity,
                "name": name,
                "confidence": round(float(confidence), 4),
                "state": state,
                "quality": quality
            })

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "success": True,
            "total_faces": len(detections),
            "detections": detections,
            "annotated_image": annotated_b64
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------------------------------------------------
# GLOBAL ERROR HANDLERS (HTTP 404 & 500)
# -------------------------------------------------------------
@app.errorhandler(404)
def handle_404_error(e):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "NOT_FOUND",
            "message": f"Resource or endpoint '{request.path}' was not found on this server."
        }), 404
    return render_template("404.html", current_user=get_current_user()), 404


@app.errorhandler(500)
def handle_500_error(e):
    import traceback
    err_tb = traceback.format_exc()
    print(f"\n[SERVER EXCEPTION 500] {e}\n{err_tb}")
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred while processing the request.",
            "details": str(e) if app.debug else "Check server logs for full audit traceback."
        }), 500
    return render_template(
        "500.html",
        current_user=get_current_user(),
        error_message=str(e) if app.debug else None
    ), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_RUN_PORT", 5000))
    active_sess = db_manager.get_active_session()
    sess_display = active_sess["session_id"] if active_sess else "None (Start in UI)"

    print(f"\n========================================================")
    print(f" SMARTCLASS AI - SEPARATE TEACHER & ADMIN PLATFORM READY")
    print(f" URL: http://127.0.0.1:{port}")
    print(f" Teacher Dashboard: http://127.0.0.1:{port}/")
    print(f" Admin Panel: http://127.0.0.1:{port}/admin")
    print(f" Active Session: {sess_display}")
    print(f"========================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
