import re
import cv2
import json
import shutil
import numpy as np
import os
import sqlite3
import datetime
import uuid
from pathlib import Path
import pandas as pd
import pymongo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "students")
SQLITE_PATH = BASE_DIR / os.getenv("SQLITE_DB_PATH", "database/students.db")
CSV_ENROLLMENT_PATH = BASE_DIR / "Students_Enrollment.csv"
PROFILES_DIR = BASE_DIR / "static" / "profiles"


class DatabaseManager:
    """
    Unified database manager supporting MongoDB with automatic SQLite fallback.
    Manages 5 core collections / tables:
      1. Students (Profiles, Department, Section, Consent, Authorized Photo)
      2. Classroom_Sessions (Academic Year, Dept, Class, Section, Subject, Faculty, Timers)
      3. Attendance (Session-based, Unique per student per session, Duration analytics)
      4. Session_Events (Check-in, Exit, Return event logs)
      5. System_Notifications (Neutral teacher notices for repeated exits)
    """

    def __init__(self):
        self.mode = "SQLite"
        self.mongo_client = None
        self.mongo_db = None
        self.is_mongo_connected = False
        self.mongo_error = None
        self.mongo_uri = MONGO_URI
        self.mongo_db_name = MONGO_DB_NAME
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        self._init_connection()
        self._init_sqlite()
        if self.is_mongo_connected:
            self._sync_sqlite_to_mongo()
        self.sync_from_csv()
        self._ensure_default_session()
        self._populate_initial_profiles()
        self.migrate_schema_and_data()

    def _init_connection(self):
        if self.mongo_uri and not self.mongo_uri.startswith(("mongodb://", "mongodb+srv://")):
            self.mongo_uri = f"mongodb://{self.mongo_uri}"

        candidates = [self.mongo_uri]
        if "127.0.0.1" in self.mongo_uri:
            candidates.append(self.mongo_uri.replace("127.0.0.1", "localhost"))
        elif "localhost" in self.mongo_uri:
            candidates.append(self.mongo_uri.replace("localhost", "127.0.0.1"))

        last_error = None
        for uri in candidates:
            try:
                client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
                client.admin.command("ping")
                self.mongo_client = client
                self.mongo_db = client[self.mongo_db_name]
                self.is_mongo_connected = True
                self.mode = "MongoDB"
                self.mongo_error = None
                self.mongo_uri = uri

                self.mongo_db.Students.create_index("roll_number", unique=True)
                self.mongo_db.Students.create_index([("degree", 1), ("stream", 1), ("academic_year", 1), ("academic_session", 1)])
                self.mongo_db.Student_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Student_Face_Embeddings.create_index("student_id")
                self.mongo_db.Faculty.create_index("faculty_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("faculty_id")
                self.mongo_db.Classroom_Sessions.create_index("session_id", unique=True)
                self.mongo_db.Attendance.create_index([("session_id", 1), ("roll_number", 1)], unique=True)
                self.mongo_db.Session_Events.create_index([("session_id", 1), ("roll_number", 1)])
                self.mongo_db.System_Notifications.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Recognition_Logs.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Classes.create_index("class_id", unique=True)
                self.mongo_db.Subjects.create_index("subject_id", unique=True)
                self.mongo_db.Teachers.create_index("teacher_id", unique=True)
                self.mongo_db.Users.create_index("username", unique=True)

                print(f"[DatabaseManager] Connected to MongoDB at {uri} (DB: {MONGO_DB_NAME}) with unique indexes.")
                return
            except Exception as e:
                last_error = e

        self.mongo_client = None
        self.mongo_db = None
        self.is_mongo_connected = False
        self.mongo_error = str(last_error)
        self.mode = "SQLite"
        print(f"[DatabaseManager] MongoDB not reachable ({last_error}). Falling back to local SQLite at {SQLITE_PATH}.")

    def _init_sqlite(self):
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()

            # 1. Students Collection Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT DEFAULT 'Computer Science',
                    section TEXT DEFAULT 'Section A',
                    profile_photo TEXT,
                    enrollment_status TEXT DEFAULT 'Active',
                    consent_given INTEGER DEFAULT 1,
                    consent_date TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Ensure columns exist if table was created in an older schema
            cursor.execute("PRAGMA table_info(students)")
            existing_cols = {col[1] for col in cursor.fetchall()}
            for col_name, col_type in [
                ("student_id", "TEXT DEFAULT ''"),
                ("class_name", "TEXT DEFAULT ''"),
                ("department", "TEXT DEFAULT 'Computer Science & Engineering'"),
                ("section", "TEXT DEFAULT ''"),
                ("profile_photo", "TEXT"),
                ("enrollment_status", "TEXT DEFAULT 'Active'"),
                ("consent_given", "INTEGER DEFAULT 1"),
                ("consent_date", "TEXT"),
                ("degree", "TEXT DEFAULT 'B.Tech'"),
                ("stream", "TEXT DEFAULT 'Computer Science & Engineering'"),
                ("academic_year", "TEXT DEFAULT '1st Year'"),
                ("academic_session", "TEXT DEFAULT '2026-27'"),
                ("email", "TEXT DEFAULT ''"),
                ("mobile", "TEXT DEFAULT ''"),
                ("face_enrollment_status", "TEXT DEFAULT 'PENDING'")
            ]:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE students ADD COLUMN {col_name} {col_type}")

            # 2. Classroom Sessions Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classroom_sessions (
                    session_id TEXT PRIMARY KEY,
                    degree TEXT DEFAULT 'B.Tech',
                    stream TEXT DEFAULT 'Computer Science & Engineering',
                    academic_year TEXT NOT NULL,
                    academic_session TEXT DEFAULT '2026-27',
                    department TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    section TEXT DEFAULT '',
                    subject TEXT NOT NULL,
                    faculty_name TEXT NOT NULL,
                    faculty_id TEXT,
                    eligible_student_ids TEXT DEFAULT '[]',
                    session_start_time TEXT NOT NULL,
                    session_end_time TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("PRAGMA table_info(classroom_sessions)")
            existing_sess_cols = {col[1] for col in cursor.fetchall()}
            for col_name, col_type in [
                ("degree", "TEXT DEFAULT 'B.Tech'"),
                ("stream", "TEXT DEFAULT 'Computer Science & Engineering'"),
                ("academic_session", "TEXT DEFAULT '2026-27'"),
                ("faculty_id", "TEXT"),
                ("eligible_student_ids", "TEXT DEFAULT '[]'")
            ]:
                if col_name not in existing_sess_cols:
                    cursor.execute(f"ALTER TABLE classroom_sessions ADD COLUMN {col_name} {col_type}")

            # 3. Attendance Table with UNIQUE(session_id, roll_number)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    attendance_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    name TEXT NOT NULL,
                    department TEXT,
                    section TEXT,
                    status TEXT DEFAULT 'PRESENT',
                    first_check_in TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_duration_minutes REAL DEFAULT 0.0,
                    total_duration_formatted TEXT DEFAULT '0m 0s',
                    exit_count INTEGER DEFAULT 0,
                    verification_method TEXT DEFAULT 'AI_FACE_RECOGNITION',
                    UNIQUE(session_id, roll_number)
                )
            """)

            # 4. Session Events Table (Entry, Exit, Return)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    details TEXT,
                    verified INTEGER DEFAULT 1
                )
            """)

            # 5. System Notifications Table (Neutral notices for repeated exits)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_notifications (
                    notification_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'NEUTRAL_NOTICE',
                    timestamp TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0
                )
            """)

            # 6. Legacy compatibility tables for Hindi & English
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hindi_attendance (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    attendance_count INTEGER DEFAULT 0,
                    last_marked TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS english_attendance (
                    roll_number TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    attendance_count INTEGER DEFAULT 0,
                    last_marked TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL
                )
            """)

            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # 7. Users Table (Role-based access: ADMIN, TEACHER)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # 8. Authorized Classes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS classes (
                    class_id TEXT PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    section TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 9. Authorized Subjects Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subjects (
                    subject_id TEXT PRIMARY KEY,
                    subject_code TEXT,
                    subject_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # 10. Authorized Teachers Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS teachers (
                    teacher_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # 11. Student Face Embeddings Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_face_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    model_version TEXT DEFAULT 'FaceNet-128D',
                    quality_score REAL DEFAULT 0.0,
                    blur_score REAL DEFAULT 0.0,
                    is_valid INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                )
            """)

            # 12. Faculty Management Table (Strict namespace separation)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faculty (
                    faculty_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    department TEXT NOT NULL,
                    designation TEXT DEFAULT 'Assistant Professor',
                    phone TEXT,
                    profile_photo TEXT,
                    face_enrollment_status TEXT DEFAULT 'PENDING',
                    created_at TEXT NOT NULL
                )
            """)

            # 13. Faculty Face Embeddings Table (Isolated namespace)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS faculty_face_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    faculty_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    model_version TEXT DEFAULT 'FaceNet-128D',
                    created_at TEXT NOT NULL
                )
            """)

            # 14. Real-Time Recognition Decisions & Audit Logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    log_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    predicted_identity_type TEXT NOT NULL,
                    matched_identity_id TEXT,
                    matched_name TEXT,
                    similarity_score REAL,
                    decision TEXT NOT NULL,
                    rejection_reason TEXT,
                    model_version TEXT DEFAULT 'FaceNet-128D',
                    timestamp TEXT NOT NULL
                )
            """)


            # 15. Student Images Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS student_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    file_hash TEXT NOT NULL,
                    mime_type TEXT DEFAULT 'image/jpeg',
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    quality_score REAL DEFAULT 0.0,
                    is_corrupted INTEGER DEFAULT 0,
                    is_duplicate INTEGER DEFAULT 0,
                    is_low_light INTEGER DEFAULT 0,
                    is_blurred INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_student_images_student_id ON student_images(student_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_student_images_file_hash ON student_images(file_hash)')

            # 16. Datasets Table (Non-identifying classroom tasks)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    task_type TEXT NOT NULL DEFAULT 'person_detection',
                    total_images INTEGER DEFAULT 0,
                    total_size_bytes INTEGER DEFAULT 0,
                    train_split REAL DEFAULT 0.70,
                    val_split REAL DEFAULT 0.15,
                    test_split REAL DEFAULT 0.15,
                    status TEXT DEFAULT 'Ready',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_datasets_task_type ON datasets(task_type)')

            # 17. Dataset Files Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_files (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER DEFAULT 0,
                    width INTEGER DEFAULT 0,
                    height INTEGER DEFAULT 0,
                    split_category TEXT DEFAULT 'train',
                    quality_score REAL DEFAULT 0.0,
                    is_corrupted INTEGER DEFAULT 0,
                    is_duplicate INTEGER DEFAULT 0,
                    is_low_light INTEGER DEFAULT 0,
                    is_blurred INTEGER DEFAULT 0,
                    annotation_data TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_files_dataset_id ON dataset_files(dataset_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dataset_files_hash ON dataset_files(file_hash)')

            # 18. Training Jobs Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    model_architecture TEXT DEFAULT 'MobileNet-PersonDetector',
                    status TEXT DEFAULT 'pending',
                    total_epochs INTEGER DEFAULT 10,
                    current_epoch INTEGER DEFAULT 0,
                    batch_size INTEGER DEFAULT 16,
                    learning_rate REAL DEFAULT 0.001,
                    optimizer TEXT DEFAULT 'adam',
                    train_loss REAL DEFAULT 0.0,
                    val_loss REAL DEFAULT 0.0,
                    train_acc REAL DEFAULT 0.0,
                    val_acc REAL DEFAULT 0.0,
                    precision REAL DEFAULT 0.0,
                    recall REAL DEFAULT 0.0,
                    mAP REAL DEFAULT 0.0,
                    progress_percent REAL DEFAULT 0.0,
                    model_version TEXT DEFAULT 'v1.0',
                    checkpoint_path TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    metrics_history TEXT DEFAULT '[]',
                    started_at TEXT,
                    completed_at TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)')

            # 19. Training Logs Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    epoch INTEGER DEFAULT 0,
                    log_level TEXT DEFAULT 'INFO',
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_logs_job_id ON training_logs(job_id)')

            # 20. Model Versions Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_versions (
                    id TEXT PRIMARY KEY,
                    version_tag TEXT NOT NULL,
                    job_id TEXT,
                    task_type TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    weights_path TEXT NOT NULL,
                    metrics TEXT DEFAULT '{}',
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_task ON model_versions(task_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active)')

            # 21. Camera Devices Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera_devices (
                    id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    device_index INTEGER DEFAULT 0,
                    resolution TEXT DEFAULT '1280x720',
                    stream_url TEXT DEFAULT '',
                    location TEXT DEFAULT 'Classroom Main',
                    status TEXT DEFAULT 'ONLINE',
                    fps REAL DEFAULT 0.0,
                    latency_ms INTEGER DEFAULT 0,
                    active_model_version TEXT DEFAULT 'v1.0-PersonOccupancy',
                    confidence_threshold REAL DEFAULT 0.50,
                    created_at TEXT NOT NULL
                )
            ''')

            # 22. Attendance Sessions Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    session_id TEXT PRIMARY KEY,
                    degree TEXT DEFAULT 'B.Tech',
                    stream TEXT DEFAULT 'Computer Science & Engineering',
                    academic_year TEXT NOT NULL,
                    academic_session TEXT DEFAULT '2026-27',
                    department TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    section TEXT DEFAULT '',
                    subject TEXT NOT NULL,
                    faculty_name TEXT NOT NULL,
                    faculty_id TEXT,
                    status TEXT DEFAULT 'ACTIVE',
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    occupancy_count INTEGER DEFAULT 0,
                    total_enrolled INTEGER DEFAULT 0,
                    verified_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_att_sessions_status ON attendance_sessions(status)')

            # 23. Attendance Records Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance_records (
                    attendance_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PRESENT',
                    first_check_in TEXT NOT NULL,
                    last_seen TEXT,
                    total_duration_minutes INTEGER DEFAULT 0,
                    verification_type TEXT DEFAULT 'TEACHER_REVIEW',
                    verified_by TEXT,
                    audit_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, student_id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_att_records_session ON attendance_records(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_att_records_student ON attendance_records(student_id)')

            # Seed default camera device if empty
            cursor.execute("SELECT COUNT(*) FROM camera_devices")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO camera_devices (id, device_name, device_index, resolution, location, status, active_model_version, confidence_threshold, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', ("CAM_01", "Classroom Main Cam", 0, "1280x720", "Hall 101 - Textile & Engineering", "ONLINE", "v1.0-PersonDetector", 0.50, now_str))

            # Seed baseline dataset if empty
            cursor.execute("SELECT COUNT(*) FROM datasets")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO datasets (id, name, description, task_type, total_images, total_size_bytes, train_split, val_split, test_split, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', ("DS_CLASSROOM_BASE", "Classroom Occupancy & Non-Identifying People Baseline", "Baseline dataset for person detection and visual occupancy headcount", "person_detection", 0, 0, 0.70, 0.15, 0.15, "Ready", now_str, now_str))

            # Seed baseline model version if empty
            cursor.execute("SELECT COUNT(*) FROM model_versions")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO model_versions (id, version_tag, job_id, task_type, architecture, weights_path, metrics, is_active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', ("MOD_v1.0_BASELINE", "v1.0", None, "person_detection", "MobileNet-PersonDetector", "Model/versions/v1.0_baseline.json", json.dumps({"val_loss": 0.22, "precision": 0.94, "recall": 0.91, "mAP": 0.93}), 1, now_str))

            # Seed default users if empty
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.executemany("""
                    INSERT INTO users (username, password, role, name, email, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    ("admin", "admin123", "ADMIN", "Administrator", "admin@smartclassroom.edu", now_str),
                    ("teacher", "teacher123", "TEACHER", "Prof. Sharma", "sharma@smartclassroom.edu", now_str)
                ])

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Seed default classes including Textile branches
            classes_to_seed = [
                ("CLS_TT_01", "B.Tech Year 3", "Textile Technology", "Section A", now_str),
                ("CLS_TT_02", "B.Tech Year 4", "Textile Technology", "Section B", now_str),
                ("CLS_TC_01", "B.Tech Year 3", "Textile Chemistry", "Section A", now_str),
                ("CLS_TC_02", "B.Tech Year 4", "Textile Chemistry", "Section B", now_str),
                ("CLS_TE_01", "B.Tech Year 3", "Textile Engineering", "Section A", now_str),
                ("CLS_TE_02", "B.Tech Year 4", "Textile Engineering", "Section B", now_str),
                ("CLS_01", "B.Tech Year 3", "Computer Science & Engineering", "Section A", now_str),
                ("CLS_02", "B.Tech Year 3", "Information Technology", "Section B", now_str),
                ("CLS_03", "B.Tech Year 2", "Artificial Intelligence & Data Science", "Section A", now_str)
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO classes (class_id, class_name, department, section, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, classes_to_seed)

            # Seed default subjects including Textile curricula
            subjects_to_seed = [
                ("SUB_TT_01", "TT301", "Yarn Manufacturing Technology", "Textile Technology", now_str),
                ("SUB_TT_02", "TT302", "Fabric Structure & Design", "Textile Technology", now_str),
                ("SUB_TC_01", "TC301", "Textile Chemical Processing & Dyeing", "Textile Chemistry", now_str),
                ("SUB_TC_02", "TC302", "Coloration & Finishing Technology", "Textile Chemistry", now_str),
                ("SUB_TE_01", "TE301", "Textile Machinery & Mechanics", "Textile Engineering", now_str),
                ("SUB_TE_02", "TE302", "Technical Textiles & Nanotechnology", "Textile Engineering", now_str),
                ("SUB_01", "CS301", "Machine Learning & AI", "Computer Science & Engineering", now_str),
                ("SUB_02", "IT302", "Robotics & Vision", "Information Technology", now_str),
                ("SUB_03", "CS303", "Deep Learning & Neural Networks", "Computer Science & Engineering", now_str),
                ("SUB_04", "CS304", "Computer Networks", "Computer Science & Engineering", now_str),
                ("SUB_05", "HUM101", "Hindi Communication", "Humanities", now_str),
                ("SUB_06", "HUM102", "English Communication", "Humanities", now_str)
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO subjects (subject_id, subject_code, subject_name, department, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, subjects_to_seed)

            # Seed default teachers including Textile faculties
            teachers_to_seed = [
                ("TCH_TT_01", "Dr. Vikram Sarabhai", "Textile Technology", "sarabhai@textile.edu", "9876543220", now_str),
                ("TCH_TC_01", "Prof. P. C. Ray", "Textile Chemistry", "pcray@textile.edu", "9876543221", now_str),
                ("TCH_TE_01", "Dr. M. Visvesvaraya", "Textile Engineering", "visvesvaraya@textile.edu", "9876543222", now_str),
                ("TCH_01", "Prof. Sharma", "Computer Science & Engineering", "sharma@smartclassroom.edu", "9876543210", now_str),
                ("TCH_02", "Dr. Alan Turing", "Computer Science & Engineering", "turing@smartclassroom.edu", "9876543211", now_str),
                ("TCH_03", "Prof. Ada Lovelace", "Information Technology", "lovelace@smartclassroom.edu", "9876543212", now_str)
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO teachers (teacher_id, name, department, email, phone, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, teachers_to_seed)

            conn.commit()

        if self.is_mongo_connected:
            try:
                for c_row in classes_to_seed:
                    self.mongo_db.Classes.update_one(
                        {"class_id": c_row[0]},
                        {"$set": {"class_id": c_row[0], "class_name": c_row[1], "department": c_row[2], "section": c_row[3], "created_at": c_row[4]}},
                        upsert=True
                    )
                for s_row in subjects_to_seed:
                    self.mongo_db.Subjects.update_one(
                        {"subject_id": s_row[0]},
                        {"$set": {"subject_id": s_row[0], "subject_code": s_row[1], "subject_name": s_row[2], "department": s_row[3], "created_at": s_row[4]}},
                        upsert=True
                    )
                for t_row in teachers_to_seed:
                    self.mongo_db.Teachers.update_one(
                        {"teacher_id": t_row[0]},
                        {"$set": {"teacher_id": t_row[0], "name": t_row[1], "department": t_row[2], "email": t_row[3], "phone": t_row[4], "created_at": t_row[5]}},
                        upsert=True
                    )
                for u_row in [
                    ("admin", "admin123", "ADMIN", "Administrator", "admin@smartclassroom.edu", now_str),
                    ("teacher", "teacher123", "TEACHER", "Prof. Sharma", "sharma@smartclassroom.edu", now_str)
                ]:
                    self.mongo_db.Users.update_one(
                        {"username": u_row[0]},
                        {"$set": {"username": u_row[0], "password": u_row[1], "role": u_row[2], "name": u_row[3], "email": u_row[4], "created_at": u_row[5]}},
                        upsert=True
                    )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB seed sync notice: {e}")

    def _ensure_default_session(self):
        """Creates an active default classroom session if none exists."""
        active = self.get_active_session()
        if not active:
            now = datetime.datetime.now()
            self.create_session(
                academic_year=f"{now.year}-{now.year + 1}",
                department="Computer Science & Engineering",
                class_name="B.Tech Year 3",
                section="Section A",
                subject="Machine Learning & AI",
                faculty_name="Prof. Sharma"
            )

    def _populate_initial_profiles(self):
        """Generates authorized profile photos from people/ dataset if not yet copied."""
        people_dir = BASE_DIR / "people"
        if not people_dir.exists():
            return

        import shutil
        for folder in people_dir.iterdir():
            if folder.is_dir():
                # Extract roll
                import re
                match = re.match(r"^(\d+)[_\s-]*(.*)$", folder.name)
                roll = match.group(1) if match else folder.name
                dest = PROFILES_DIR / f"{roll}.jpg"
                if not dest.exists():
                    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
                    if images:
                        shutil.copyfile(str(images[0]), str(dest))
                        with sqlite3.connect(SQLITE_PATH) as conn:
                            conn.cursor().execute(
                                "UPDATE students SET profile_photo = ? WHERE roll_number = ?",
                                (f"/static/profiles/{roll}.jpg", roll)
                            )
                            conn.commit()

    def try_auto_reconnect(self):
        """Opportunistically attempts to reconnect to MongoDB if it came online after startup."""
        if self.is_mongo_connected and self.mongo_client is not None:
            return True
        if self.mongo_uri and not self.mongo_uri.startswith(("mongodb://", "mongodb+srv://")):
            self.mongo_uri = f"mongodb://{self.mongo_uri}"

        candidates = [self.mongo_uri]
        if "127.0.0.1" in self.mongo_uri:
            candidates.append(self.mongo_uri.replace("127.0.0.1", "localhost"))
        elif "localhost" in self.mongo_uri:
            candidates.append(self.mongo_uri.replace("localhost", "127.0.0.1"))

        for uri in candidates:
            try:
                client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=1500, connectTimeoutMS=1500)
                client.admin.command("ping")
                self.mongo_client = client
                self.mongo_db = client[self.mongo_db_name]
                self.is_mongo_connected = True
                self.mode = "MongoDB"
                self.mongo_error = None
                self.mongo_uri = uri

                self.mongo_db.Students.create_index("roll_number", unique=True)
                self.mongo_db.Students.create_index([("degree", 1), ("stream", 1), ("academic_year", 1), ("academic_session", 1)])
                self.mongo_db.Student_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Student_Face_Embeddings.create_index("student_id")
                self.mongo_db.Faculty.create_index("faculty_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("faculty_id")
                self.mongo_db.Classroom_Sessions.create_index("session_id", unique=True)
                self.mongo_db.Attendance.create_index([("session_id", 1), ("roll_number", 1)], unique=True)
                self.mongo_db.Session_Events.create_index([("session_id", 1), ("roll_number", 1)])
                self.mongo_db.System_Notifications.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Recognition_Logs.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Classes.create_index("class_id", unique=True)
                self.mongo_db.Subjects.create_index("subject_id", unique=True)
                self.mongo_db.Teachers.create_index("teacher_id", unique=True)
                self.mongo_db.Users.create_index("username", unique=True)

                self._sync_sqlite_to_mongo()
                print(f"[DatabaseManager] Auto-reconnected to MongoDB at {uri} (DB: {self.mongo_db_name}) and synchronized data.")
                return True
            except Exception:
                continue
        return False

    def get_status(self):
        if not self.is_mongo_connected:
            self.try_auto_reconnect()

        current_db = getattr(self, "mongo_db_name", MONGO_DB_NAME)
        return {
            "mode": self.mode,
            "is_mongo_connected": self.is_mongo_connected,
            "mongo_uri": self.mongo_uri,
            "mongo_db": current_db,
            "database_name": current_db,
            "sqlite_path": str(SQLITE_PATH),
            "mongo_error": self.mongo_error,
            "collections": [
                "Students",
                "Student_Face_Embeddings",
                "Faculty",
                "Faculty_Face_Embeddings",
                "Classroom_Sessions",
                "Attendance",
                "Session_Events",
                "System_Notifications",
                "Recognition_Logs",
                "Classes",
                "Subjects",
                "Teachers",
                "Users",
                "Hindi",
                "English"
            ],
            "status_message": (
                f"Connected to MongoDB ({current_db})"
                if self.is_mongo_connected
                else f"Active: SQLite Local Storage ({SQLITE_PATH.name}) - MongoDB Offline"
            ),
        }

    def test_or_reconnect_mongo(self, mongo_uri=None, db_name=None):
        target_uri = mongo_uri or self.mongo_uri
        target_db = db_name or getattr(self, "mongo_db_name", MONGO_DB_NAME)

        if target_uri and not target_uri.startswith(("mongodb://", "mongodb+srv://")):
            target_uri = f"mongodb://{target_uri}"

        candidates = [target_uri]
        if "127.0.0.1" in target_uri:
            candidates.append(target_uri.replace("127.0.0.1", "localhost"))
        elif "localhost" in target_uri:
            candidates.append(target_uri.replace("localhost", "127.0.0.1"))

        last_err = None
        for uri in candidates:
            try:
                client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
                client.admin.command("ping")
                self.mongo_client = client
                self.mongo_db = client[target_db]
                self.is_mongo_connected = True
                self.mode = "MongoDB"
                self.mongo_error = None
                self.mongo_uri = uri
                self.mongo_db_name = target_db

                self.mongo_db.Students.create_index("roll_number", unique=True)
                self.mongo_db.Students.create_index([("degree", 1), ("stream", 1), ("academic_year", 1), ("academic_session", 1)])
                self.mongo_db.Student_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Student_Face_Embeddings.create_index("student_id")
                self.mongo_db.Faculty.create_index("faculty_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("embedding_id", unique=True)
                self.mongo_db.Faculty_Face_Embeddings.create_index("faculty_id")
                self.mongo_db.Classroom_Sessions.create_index("session_id", unique=True)
                self.mongo_db.Attendance.create_index([("session_id", 1), ("roll_number", 1)], unique=True)
                self.mongo_db.Session_Events.create_index([("session_id", 1), ("roll_number", 1)])
                self.mongo_db.System_Notifications.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Recognition_Logs.create_index([("session_id", 1), ("timestamp", -1)])
                self.mongo_db.Classes.create_index("class_id", unique=True)
                self.mongo_db.Subjects.create_index("subject_id", unique=True)
                self.mongo_db.Teachers.create_index("teacher_id", unique=True)
                self.mongo_db.Users.create_index("username", unique=True)

                self._sync_sqlite_to_mongo()
                print(f"[DatabaseManager] Successfully reconnected to MongoDB at {uri} (DB: {target_db})")
                return True, f"Successfully connected to MongoDB '{target_db}' at {uri}."
            except Exception as e:
                last_err = e

        self.mongo_error = str(last_err)
        return False, f"MongoDB connection failed: {last_err}"

    def _sync_sqlite_to_mongo(self):
        if not self.is_mongo_connected or self.mongo_db is None:
            return
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Sync students
                cursor.execute("SELECT * FROM students")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Students.update_one(
                        {"roll_number": d["roll_number"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync sessions
                cursor.execute("SELECT * FROM classroom_sessions")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Classroom_Sessions.update_one(
                        {"session_id": d["session_id"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync attendance
                cursor.execute("SELECT * FROM attendance")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Attendance.update_one(
                        {"session_id": d["session_id"], "roll_number": d["roll_number"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync classes
                cursor.execute("SELECT * FROM classes")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Classes.update_one(
                        {"class_id": d["class_id"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync subjects
                cursor.execute("SELECT * FROM subjects")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Subjects.update_one(
                        {"subject_id": d["subject_id"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync teachers
                cursor.execute("SELECT * FROM teachers")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Teachers.update_one(
                        {"teacher_id": d["teacher_id"]},
                        {"$set": d},
                        upsert=True
                    )

                # Sync users
                cursor.execute("SELECT * FROM users")
                for row in cursor.fetchall():
                    d = dict(row)
                    self.mongo_db.Users.update_one(
                        {"username": d["username"]},
                        {"$set": d},
                        upsert=True
                    )
        except Exception as e:
            print(f"[DatabaseManager] Sync to Mongo error: {e}")

    # =========================================================================
    # 1. CLASSROOM SESSION MANAGEMENT (Class-Group Scoped Attendance)
    # =========================================================================
    def create_session(self, degree_or_year="B.Tech", stream_or_dept="Computer Science & Engineering",
                       academic_year_or_class="2nd Year", academic_session_or_sec="2026-27",
                       subject="Machine Learning & AI", faculty_name="Prof. Sharma",
                       faculty_id=None, section="", department=None, class_name=None,
                       degree=None, stream=None, academic_year=None, academic_session=None):
        """
        Creates a new unique classroom session bound to an immutable class-group snapshot:
        Degree + Stream/Branch + Academic Year + Academic Session.
        Filters eligible student roster before matching.
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Normalize whether called with keyword/modern args or legacy positional
        if degree is not None or stream is not None or academic_year is not None or academic_session is not None:
            deg = degree or "B.Tech"
            strm = stream or department or "Computer Science & Engineering"
            ayear = academic_year or "2nd Year"
            asess = academic_session or "2026-27"
            subj = subject
            fname = faculty_name
            sec = section or ""
        elif "-" in str(degree_or_year) and any(c.isalpha() for c in str(stream_or_dept)):
            # Legacy positional: (academic_year, department, class_name, section, subject, faculty_name)
            asess = str(degree_or_year)
            strm = str(stream_or_dept)
            cls_n = str(academic_year_or_class)
            sec = str(academic_session_or_sec)
            subj = subject
            fname = faculty_name
            ayear = "2nd Year"
            deg = "B.Tech" if "b.tech" in cls_n.lower() else ("M.Tech" if "m.tech" in cls_n.lower() else "B.Tech")
        else:
            deg = str(degree_or_year or "B.Tech")
            strm = str(stream_or_dept or "Computer Science & Engineering")
            ayear = str(academic_year_or_class or "2nd Year")
            asess = str(academic_session_or_sec or "2026-27")
            subj = subject
            fname = faculty_name
            sec = section or ""

        dept = department or strm
        cls_name = class_name or f"{deg} {ayear}"
        clean_stream = re.sub(r"[^\w]", "", strm)[:8].upper()
        clean_subj = re.sub(r"[^\w]", "", subj)[:6].upper()
        unique_suffix = uuid.uuid4().hex[:4].upper()
        session_id = f"SESS_{now.strftime('%Y%m%d_%H%M%S')}_{clean_stream}_{clean_subj}_{unique_suffix}"

        # End any currently active session
        self.end_active_sessions()

        # Build immutable snapshot of eligible students strictly matching this class group
        all_students = self.get_all_students()
        eligible_students = [
            s for s in all_students
            if s.get("degree", "").strip().lower() == deg.strip().lower()
            and s.get("stream", "").strip().lower() == strm.strip().lower()
            and s.get("academic_year", "").strip().lower() == ayear.strip().lower()
            and s.get("academic_session", "").strip().lower() == asess.strip().lower()
        ]
        # Secondary fallback: match legacy records where degree/session might be default or unassigned
        if not eligible_students:
            eligible_students = [
                s for s in all_students
                if (s.get("stream", "").strip().lower() == strm.strip().lower() or s.get("department", "").strip().lower() == dept.strip().lower())
                and (not s.get("academic_year") or s.get("academic_year", "").strip().lower() == ayear.strip().lower())
                and (not s.get("degree") or s.get("degree", "").strip().lower() == deg.strip().lower())
                and (not s.get("academic_session") or s.get("academic_session", "").strip().lower() == asess.strip().lower())
            ]

        eligible_ids = [str(s["roll_number"]).strip() for s in eligible_students]

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO classroom_sessions (
                    session_id, degree, stream, academic_year, academic_session,
                    department, class_name, section, subject, faculty_name,
                    faculty_id, eligible_student_ids, session_start_time,
                    status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)
            """, (
                session_id, deg, strm, ayear, asess,
                dept, cls_name, sec, subj, fname,
                faculty_id or "", json.dumps(eligible_ids), timestamp, timestamp
            ))
            cursor.execute(
                "INSERT INTO audit_logs (action, target, details, timestamp) VALUES (?, ?, ?, ?)",
                ("CREATE_SESSION", session_id, f"{deg} {strm} ({ayear} {asess}) - {subj} by {fname} [Eligible: {len(eligible_ids)}]", timestamp)
            )
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Classroom_Sessions.insert_one({
                    "session_id": session_id,
                    "degree": deg,
                    "stream": strm,
                    "academic_year": ayear,
                    "academic_session": asess,
                    "department": dept,
                    "class_name": cls_name,
                    "section": sec,
                    "subject": subj,
                    "faculty_name": fname,
                    "faculty_id": faculty_id or "",
                    "eligible_student_ids": eligible_ids,
                    "session_start_time": timestamp,
                    "session_end_time": None,
                    "status": "ACTIVE",
                    "created_at": timestamp
                })
            except Exception as e:
                print(f"[DatabaseManager] MongoDB session creation error: {e}")

        return session_id

    def get_eligible_students_for_session(self, session_id):
        """Returns the list of eligible student roll numbers snapshotted for this session."""
        sess = self.get_session(session_id)
        if not sess:
            return []
        ids = sess.get("eligible_student_ids")
        if isinstance(ids, str):
            try:
                return json.loads(ids)
            except Exception:
                return []
        elif isinstance(ids, list):
            return ids
        return []

    def get_active_session(self):
        """Retrieves the current active session record."""
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classroom_sessions WHERE status = 'ACTIVE' ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def end_active_sessions(self):
        """Marks any currently active sessions as COMPLETED."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE classroom_sessions SET status = 'COMPLETED', session_end_time = ? WHERE status = 'ACTIVE'", (now,))
            conn.commit()

        if self.is_mongo_connected:
            try:
                self.mongo_db.Classroom_Sessions.update_many(
                    {"status": "ACTIVE"},
                    {"$set": {"status": "COMPLETED", "session_end_time": now}}
                )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB session end error: {e}")

    def get_all_sessions(self, limit=20):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classroom_sessions ORDER BY created_at DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_session(self, session_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM classroom_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # =========================================================================
    # 2. STUDENT PROFILE MANAGEMENT (No Section Requirement)
    # =========================================================================
    def enroll_student(self, roll_number, name, email="", mobile="",
                       degree="B.Tech", stream="Computer Science & Engineering",
                       academic_year="1st Year", academic_session="2026-27",
                       department=None, section="", profile_photo=None,
                       sync_csv=True, consent_given=1, face_enrollment_status="PENDING"):
        """
        Enrolls student with full academic metadata:
        Full Name, Enrollment Number / Student ID, Email, Mobile, Degree, Stream, Year, Session.
        The Section field is completely optional and non-mandatory.
        """
        roll = str(roll_number).strip()
        student_name = str(name).strip()
        if not roll or not student_name:
            return False, "Roll number and student name cannot be empty."

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photo_url = profile_photo or f"/static/profiles/{roll}.jpg"
        consent_val = 1 if consent_given else 0
        dept = department or stream or "Computer Science & Engineering"
        sec = section or ""

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (
                    roll_number, name, email, mobile, degree, stream, academic_year, academic_session,
                    department, section, profile_photo, enrollment_status,
                    consent_given, consent_date, face_enrollment_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Active', ?, ?, ?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET
                    name=excluded.name, email=excluded.email, mobile=excluded.mobile,
                    degree=excluded.degree, stream=excluded.stream, academic_year=excluded.academic_year,
                    academic_session=excluded.academic_session, department=excluded.department,
                    section=excluded.section, profile_photo=excluded.profile_photo,
                    consent_given=excluded.consent_given, consent_date=excluded.consent_date,
                    face_enrollment_status=COALESCE(excluded.face_enrollment_status, students.face_enrollment_status)
            """, (
                roll, student_name, email, mobile, degree, stream, academic_year, academic_session,
                dept, sec, photo_url, consent_val, now, face_enrollment_status, now
            ))

            # Maintain legacy compatibility tables
            cursor.execute("INSERT OR IGNORE INTO hindi_attendance (roll_number, name, attendance_count) VALUES (?, ?, 0)", (roll, student_name))
            cursor.execute("INSERT OR IGNORE INTO english_attendance (roll_number, name, attendance_count) VALUES (?, ?, 0)", (roll, student_name))
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Students.update_one(
                    {"roll_number": roll},
                    {"$set": {
                        "name": student_name,
                        "email": email,
                        "mobile": mobile,
                        "degree": degree,
                        "stream": stream,
                        "academic_year": academic_year,
                        "academic_session": academic_session,
                        "department": dept,
                        "section": sec,
                        "profile_photo": photo_url,
                        "enrollment_status": "Active",
                        "consent_given": bool(consent_val),
                        "consent_date": now,
                        "face_enrollment_status": face_enrollment_status,
                        "updated_at": now
                    }, "$setOnInsert": {"created_at": now}},
                    upsert=True
                )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB student profile error: {e}")

        if sync_csv:
            self._update_enrollment_csv(roll, student_name, department=dept, section=sec, profile_photo=photo_url, consent_given=consent_val)

        return True, f"Student {student_name} (Roll: {roll}) enrolled successfully in {degree} {stream}."

    def update_student_face_status(self, roll_number, status):
        """Updates the face enrollment status badge for a student."""
        roll = str(roll_number).strip()
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.cursor().execute("UPDATE students SET face_enrollment_status = ? WHERE roll_number = ?", (status, roll))
                conn.commit()
        except Exception:
            pass
        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Students.update_one({"roll_number": roll}, {"$set": {"face_enrollment_status": status}})
            except Exception:
                pass

    def _normalize_student(self, d):
        if not d:
            return d
        if not d.get("degree"): d["degree"] = "B.Tech"
        if not d.get("stream"): d["stream"] = d.get("department") or "Computer Science & Engineering"
        if not d.get("academic_year"): d["academic_year"] = "1st Year"
        if not d.get("academic_session"): d["academic_session"] = "2026-27"
        if not d.get("email"): d["email"] = ""
        if not d.get("mobile"): d["mobile"] = ""
        if "face_enrollment_status" not in d or not d["face_enrollment_status"]:
            folder = self._get_student_folder(d.get("roll_number"))
            if folder and any(folder.glob("*.jpg")):
                d["face_enrollment_status"] = "ENROLLED"
            else:
                d["face_enrollment_status"] = "PENDING"
        return d

    def update_student_profile(self, roll_number, name, department, section, enrollment_status):
        """Teacher/Admin correction of student profile data."""
        roll = str(roll_number).strip()
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE students SET
                    name = ?,
                    department = ?,
                    section = ?,
                    enrollment_status = ?
                WHERE roll_number = ?
            """, (name.strip(), department.strip(), section.strip(), enrollment_status.strip(), roll))
            conn.commit()

        if self.is_mongo_connected:
            try:
                self.mongo_db.Students.update_one(
                    {"roll_number": roll},
                    {"$set": {
                        "name": name.strip(),
                        "department": department.strip(),
                        "section": section.strip(),
                        "enrollment_status": enrollment_status.strip()
                    }}
                )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB profile update error: {e}")

        return True, f"Profile for Roll {roll} updated."

    def get_all_students(self, section=None, degree=None, stream=None, academic_year=None, academic_session=None):
        """Retrieve student profiles, merging MongoDB and local SQLite for 100% data reliability."""
        students_by_roll = {}
        # 1. Load from SQLite
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                query = "SELECT * FROM students WHERE 1=1"
                params = []
                if section and section != "ALL":
                    query += " AND section = ?"
                    params.append(section)
                if degree and degree != "ALL":
                    query += " AND degree = ?"
                    params.append(degree)
                if stream and stream != "ALL":
                    query += " AND (stream = ? OR department = ?)"
                    params.extend([stream, stream])
                if academic_year and academic_year != "ALL":
                    query += " AND academic_year = ?"
                    params.append(academic_year)
                if academic_session and academic_session != "ALL":
                    query += " AND academic_session = ?"
                    params.append(academic_session)

                query += " ORDER BY roll_number ASC"
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    d = dict(row)
                    students_by_roll[str(d["roll_number"]).strip()] = self._normalize_student(d)
        except Exception as e:
            print(f"[DatabaseManager] SQLite get_all_students warning: {e}")

        # 2. Merge from MongoDB if connected
        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                q = {}
                if section and section != "ALL":
                    q["section"] = section
                if degree and degree != "ALL":
                    q["degree"] = degree
                if stream and stream != "ALL":
                    q["$or"] = [{"stream": stream}, {"department": stream}]
                if academic_year and academic_year != "ALL":
                    q["academic_year"] = academic_year
                if academic_session and academic_session != "ALL":
                    q["academic_session"] = academic_session

                for doc in self.mongo_db.Students.find(q):
                    doc.pop("_id", None)
                    r = str(doc.get("roll_number", "")).strip()
                    if r:
                        norm = self._normalize_student(doc)
                        if r not in students_by_roll:
                            students_by_roll[r] = norm
                        else:
                            for k, v in norm.items():
                                if v is not None and v != "":
                                    students_by_roll[r][k] = v
            except Exception as e:
                print(f"[DatabaseManager] MongoDB get_all_students warning: {e}")

        return sorted(list(students_by_roll.values()), key=lambda s: str(s.get("roll_number", "")))

    def get_student(self, roll_number):
        roll = str(roll_number).strip()
        student = None
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM students WHERE roll_number = ?", (roll,))
                row = cursor.fetchone()
                if row:
                    student = dict(row)
        except Exception:
            pass

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                doc = self.mongo_db.Students.find_one({"roll_number": roll})
                if doc:
                    doc.pop("_id", None)
                    if not student:
                        student = doc
                    else:
                        for k, v in doc.items():
                            if v is not None and v != "":
                                student[k] = v
            except Exception:
                pass

        return self._normalize_student(student) if student else None

    def delete_student(self, roll_number):
        """Deletes student record, profile image, and all biometric image folders."""
        roll = str(roll_number).strip()
        student = self.get_student(roll)
        student_name = student["name"] if student else roll

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM attendance WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM session_events WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM system_notifications WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM hindi_attendance WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM english_attendance WHERE roll_number = ?", (roll,))
            cursor.execute("DELETE FROM attendance_logs WHERE roll_number = ?", (roll,))
            cursor.execute(
                "INSERT INTO audit_logs (action, target, details, timestamp) VALUES (?, ?, ?, ?)",
                ("DELETE_STUDENT", roll, f"Purged data for {student_name}", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()

        if self.is_mongo_connected:
            try:
                self.mongo_db.Students.delete_one({"roll_number": roll})
                self.mongo_db.Attendance.delete_many({"roll_number": roll})
                self.mongo_db.Session_Events.delete_many({"roll_number": roll})
                self.mongo_db.System_Notifications.delete_many({"roll_number": roll})
            except Exception as e:
                print(f"[DatabaseManager] MongoDB student deletion error: {e}")

        # Remove profile photo
        profile_file = PROFILES_DIR / f"{roll}.jpg"
        if profile_file.exists():
            try:
                profile_file.unlink()
            except Exception:
                pass

        # Remove image folder
        people_dir = BASE_DIR / "people"
        if people_dir.exists():
            for folder in people_dir.iterdir():
                if folder.is_dir() and (folder.name.startswith(roll) or folder.name.startswith(f"{roll}_")):
                    import shutil
                    shutil.rmtree(folder, ignore_errors=True)

        # Update CSV
        if CSV_ENROLLMENT_PATH.exists():
            try:
                df = pd.read_csv(CSV_ENROLLMENT_PATH)
                df["Roll Number"] = df["Roll Number"].astype(str)
                df = df[df["Roll Number"] != roll]
                df.to_csv(CSV_ENROLLMENT_PATH, index=False)
            except Exception:
                pass

        return True, f"Student {student_name} (Roll: {roll}) and all biometric data permanently deleted."

    def _get_student_folder(self, roll_number, create=False):
        roll = str(roll_number).strip()
        student = self.get_student(roll)
        student_name = student["name"] if student else roll
        clean_name = re.sub(r"[^\w\-_]", "", student_name.replace(" ", "_"))

        people_dir = BASE_DIR / "people"
        people_dir.mkdir(parents=True, exist_ok=True)

        for folder in people_dir.iterdir():
            if folder.is_dir() and (folder.name.startswith(f"{roll}_") or folder.name == roll):
                return folder

        if create:
            new_folder = people_dir / f"{roll}_{clean_name}"
            new_folder.mkdir(parents=True, exist_ok=True)
            return new_folder
        return None

    def get_student_photos(self, roll_number):
        """Returns list of photo records for a student, limited to up to 10 photos."""
        folder = self._get_student_folder(roll_number)
        if not folder or not folder.exists():
            return []

        photos = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and f.is_file():
                photos.append({
                    "filename": f.name,
                    "url": f"/people_photos/{folder.name}/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        return photos[:10]

    def add_student_photo(self, roll_number, image_bytes, filename=None):
        """Adds a photo to student dataset ensuring max 10 photos, validates format, updates profile."""
        roll = str(roll_number).strip()
        student = self.get_student(roll)
        if not student:
            return False, "Student not found."

        if not student.get("consent_given", 1):
            return False, "Biometric consent not authorized for this student."

        folder = self._get_student_folder(roll, create=True)
        existing = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(existing) >= 10:
            return False, "Maximum limit of 10 photographs per student reached. Delete or replace an existing photo."

        # Validate image format via OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False, "Invalid image format. Supported formats: JPEG, PNG."

        # Determine next photo filename (1.jpg to 10.jpg)
        used_numbers = []
        for f in existing:
            m = re.match(r"^(\d+)\.", f.name)
            if m:
                used_numbers.append(int(m.group(1)))

        next_num = 1
        for n in range(1, 11):
            if n not in used_numbers:
                next_num = n
                break

        dest_file = folder / f"{next_num}.jpg"
        cv2.imwrite(str(dest_file), img)

        # Update profile photo if none or first
        prof_file = PROFILES_DIR / f"{roll}.jpg"
        if not prof_file.exists() or len(existing) == 0:
            cv2.imwrite(str(prof_file), img)
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.cursor().execute("UPDATE students SET profile_photo = ? WHERE roll_number = ?", (f"/static/profiles/{roll}.jpg", roll))
                conn.commit()

        # Trigger immediate automatic face enrollment into the recognition index
        enroll_notice = ""
        try:
            from ai_engine import auto_enroll_student_face
            enroll_res = auto_enroll_student_face(roll, student.get("name", roll), img)
            if enroll_res.get("success"):
                enroll_notice = " Biometric face model enrolled and activated automatically."
            else:
                enroll_notice = f" Enrollment status: {enroll_res.get('message', 'Processing issue.')}"
        except Exception as e:
            enroll_notice = f" Enrollment notice: {e}"

        return True, f"Photo saved as {dest_file.name} for student {student['name']}.{enroll_notice}"

    def delete_student_photo(self, roll_number, filename):
        """Deletes a specific photograph from student's dataset and updates model."""
        folder = self._get_student_folder(roll_number)
        if not folder:
            return False, "Student folder not found."

        target = folder / filename
        if target.exists():
            target.unlink()
            # If remaining photos exist, re-enroll remaining; otherwise update status
            remaining = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if remaining:
                try:
                    from ai_engine import auto_enroll_student_face
                    auto_enroll_student_face(roll_number, "Student", cv2.imread(str(remaining[0])))
                except Exception:
                    pass
            else:
                self.update_student_face_status(roll_number, "PENDING")
            return True, f"Photo {filename} deleted."
        return False, "Photo not found."

    def replace_student_photo(self, roll_number, filename, image_bytes):
        """Replaces a specific photograph and updates face enrollment."""
        folder = self._get_student_folder(roll_number)
        if not folder:
            return False, "Student folder not found."

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False, "Invalid image format. Supported formats: JPEG, PNG."

        target = folder / filename
        cv2.imwrite(str(target), img)

        try:
            from ai_engine import auto_enroll_student_face
            student = self.get_student(roll_number)
            s_name = student.get("name", roll_number) if student else roll_number
            auto_enroll_student_face(roll_number, s_name, img)
        except Exception as e:
            print(f"[ReplacePhoto] Auto-enroll notice: {e}")

        return True, f"Photo {filename} replaced successfully. AI face enrollment updated."

    # =========================================================================
    # CLASS, SUBJECT, TEACHER & USER MANAGEMENT (Admin Panel)
    # =========================================================================
    def get_all_classes(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT * FROM classes ORDER BY class_name, section").fetchall()]

    def create_class(self, class_name, department, section):
        class_id = f"CLS_{uuid.uuid4().hex[:6].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute(
                "INSERT INTO classes (class_id, class_name, department, section, created_at) VALUES (?, ?, ?, ?, ?)",
                (class_id, class_name.strip(), department.strip(), section.strip(), now)
            )
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Classes.update_one(
                    {"class_id": class_id},
                    {"$set": {"class_id": class_id, "class_name": class_name.strip(), "department": department.strip(), "section": section.strip(), "created_at": now}},
                    upsert=True
                )
            except Exception:
                pass
        return True, f"Class '{class_name} ({section})' created."

    def delete_class(self, class_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("DELETE FROM classes WHERE class_id = ?", (class_id,))
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Classes.delete_one({"class_id": class_id})
            except Exception:
                pass
        return True, "Class removed."

    def get_all_subjects(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT * FROM subjects ORDER BY subject_name").fetchall()]

    def create_subject(self, subject_name, department, subject_code=None):
        sub_id = f"SUB_{uuid.uuid4().hex[:6].upper()}"
        code = subject_code.strip() if subject_code else sub_id
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute(
                "INSERT INTO subjects (subject_id, subject_code, subject_name, department, created_at) VALUES (?, ?, ?, ?, ?)",
                (sub_id, code, subject_name.strip(), department.strip(), now)
            )
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Subjects.update_one(
                    {"subject_id": sub_id},
                    {"$set": {"subject_id": sub_id, "subject_code": code, "subject_name": subject_name.strip(), "department": department.strip(), "created_at": now}},
                    upsert=True
                )
            except Exception:
                pass
        return True, f"Subject '{subject_name}' created."

    def delete_subject(self, subject_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("DELETE FROM subjects WHERE subject_id = ?", (subject_id,))
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Subjects.delete_one({"subject_id": subject_id})
            except Exception:
                pass
        return True, "Subject removed."

    def get_all_teachers(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT * FROM teachers ORDER BY name").fetchall()]

    def create_teacher(self, name, department, email=None, phone=None):
        tch_id = f"TCH_{uuid.uuid4().hex[:6].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute(
                "INSERT INTO teachers (teacher_id, name, department, email, phone, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (tch_id, name.strip(), department.strip(), email or "", phone or "", now)
            )
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Teachers.update_one(
                    {"teacher_id": tch_id},
                    {"$set": {"teacher_id": tch_id, "name": name.strip(), "department": department.strip(), "email": email or "", "phone": phone or "", "created_at": now}},
                    upsert=True
                )
            except Exception:
                pass
        return True, f"Teacher '{name}' registered."

    def delete_teacher(self, teacher_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("DELETE FROM teachers WHERE teacher_id = ?", (teacher_id,))
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Teachers.delete_one({"teacher_id": teacher_id})
            except Exception:
                pass
        return True, "Teacher removed."

    def get_all_users(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.cursor().execute("SELECT username, role, name, email, created_at FROM users ORDER BY role, username").fetchall()]

    def authenticate_user(self, username, password):
        u = str(username).strip()
        p = str(password).strip()
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT username, role, name, email FROM users WHERE username = ? AND password = ?", (u, p))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_user(self, username, password, role, name, email=None):
        u = str(username).strip().lower()
        if not u or not password:
            return False, "Username and password required."
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE username = ?", (u,))
            if cursor.fetchone():
                return False, f"User '{u}' already exists."
            cursor.execute(
                "INSERT INTO users (username, password, role, name, email, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (u, password.strip(), role.upper(), name.strip(), email or "", now)
            )
            conn.commit()
        return True, f"User '{u}' created with role {role}."

    def delete_user(self, username):
        u = str(username).strip()
        if u.lower() == "admin":
            return False, "Cannot delete primary administrator."
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("DELETE FROM users WHERE username = ?", (u,))
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.Users.delete_one({"username": u})
            except Exception:
                pass
        return True, f"User '{u}' removed."

    # =========================================================================
    # 3. SESSION ATTENDANCE, DURATION ANALYTICS & EXIT EVENT MANAGEMENT
    # =========================================================================
    def record_session_presence(self, session_id, roll_number, verification_method="AI_FACE_RECOGNITION"):
        """
        Core attendance tracking rule:
          - If student was not checked-in: marks PRESENT, logs CHECK_IN event.
          - If student was already marked PRESENT: updates last_seen and recalculates duration.
          - If student was previously CHECKED_OUT and returned: marks PRESENT, logs RETURN event.
          - CRITICAL: Never creates a duplicate attendance record for the same session.
        """
        roll = str(roll_number).strip()

        # Verify session exists and is currently active
        sess = self.get_session(session_id)
        if not sess or sess.get("status") != "ACTIVE":
            return False, "Attendance cannot be recorded for an inactive or concluded session.", "SESSION_CLOSED"

        # Verify student is in eligible roster for this session (if class scope configured)
        eligible_ids = self.get_eligible_students_for_session(session_id)
        if eligible_ids and roll not in eligible_ids:
            return False, f"Student {roll} is not registered in this class roster.", "OUTSIDE_CLASS"

        student = self.get_student(roll)
        if not student:
            return False, "Student not found in enrolled database.", None

        # Verify informed consent
        if not student.get("consent_given", 1):
            return False, f"Biometric processing not authorized for student {student['name']}.", None

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if an attendance record already exists for this (session_id, roll_number)
            cursor.execute(
                "SELECT * FROM attendance WHERE session_id = ? AND roll_number = ?",
                (session_id, roll)
            )
            existing = cursor.fetchone()

            if not existing:
                # FIRST CHECK-IN IN THIS SESSION
                att_id = f"ATT_{uuid.uuid4().hex[:8].upper()}"
                cursor.execute("""
                    INSERT INTO attendance (
                        attendance_id, session_id, roll_number, name,
                        department, section, status, first_check_in,
                        last_seen, total_duration_minutes, total_duration_formatted,
                        exit_count, verification_method
                    ) VALUES (?, ?, ?, ?, ?, ?, 'PRESENT', ?, ?, 0.0, '0m 0s', 0, ?)
                """, (
                    att_id, session_id, roll, student["name"],
                    student.get("department", ""), student.get("section", ""),
                    timestamp, timestamp, verification_method
                ))

                # Log CHECK_IN Event
                event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"
                cursor.execute("""
                    INSERT INTO session_events (
                        event_id, session_id, roll_number, name,
                        event_type, timestamp, details, verified
                    ) VALUES (?, ?, ?, ?, 'CHECK_IN', ?, 'Initial biometric check-in verified', 1)
                """, (event_id, session_id, roll, student["name"], timestamp))

                conn.commit()

                if self.is_mongo_connected:
                    try:
                        self.mongo_db.Attendance.update_one(
                            {"session_id": session_id, "roll_number": roll},
                            {"$setOnInsert": {
                                "attendance_id": att_id,
                                "session_id": session_id,
                                "roll_number": roll,
                                "name": student["name"],
                                "department": student.get("department", ""),
                                "section": student.get("section", ""),
                                "status": "PRESENT",
                                "first_check_in": timestamp,
                                "last_seen": timestamp,
                                "total_duration_minutes": 0.0,
                                "total_duration_formatted": "0m 0s",
                                "exit_count": 0,
                                "verification_method": verification_method
                            }},
                            upsert=True
                        )
                        self.mongo_db.Session_Events.insert_one({
                            "event_id": event_id,
                            "session_id": session_id,
                            "roll_number": roll,
                            "name": student["name"],
                            "event_type": "CHECK_IN",
                            "timestamp": timestamp,
                            "details": "Initial biometric check-in verified",
                            "verified": True
                        })
                    except Exception as e:
                        print(f"[DatabaseManager] MongoDB presence insert error: {e}")

                return True, f"Check-in verified for {student['name']} ({roll}).", "CHECK_IN"

            else:
                # Student already has an attendance record in this session
                current_status = existing["status"]
                first_in_dt = datetime.datetime.strptime(existing["first_check_in"], "%Y-%m-%d %H:%M:%S")
                total_seconds = max(0, (now - first_in_dt).total_seconds())
                total_min = round(total_seconds / 60.0, 1)
                formatted_dur = f"{int(total_seconds // 60)}m {int(total_seconds % 60)}s"

                if current_status == "CHECKED_OUT":
                    # STUDENT RETURNED AFTER TEMPORARY EXIT
                    event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"
                    cursor.execute("""
                        INSERT INTO session_events (
                            event_id, session_id, roll_number, name,
                            event_type, timestamp, details, verified
                        ) VALUES (?, ?, ?, ?, 'RETURN', ?, 'Student returned to classroom session', 1)
                    """, (event_id, session_id, roll, student["name"], timestamp))

                    cursor.execute("""
                        UPDATE attendance SET
                            status = 'PRESENT',
                            last_seen = ?,
                            total_duration_minutes = ?,
                            total_duration_formatted = ?
                        WHERE session_id = ? AND roll_number = ?
                    """, (timestamp, total_min, formatted_dur, session_id, roll))
                    conn.commit()

                    if self.is_mongo_connected:
                        try:
                            self.mongo_db.Attendance.update_one(
                                {"session_id": session_id, "roll_number": roll},
                                {"$set": {
                                    "status": "PRESENT",
                                    "last_seen": timestamp,
                                    "total_duration_minutes": total_min,
                                    "total_duration_formatted": formatted_dur
                                }}
                            )
                            self.mongo_db.Session_Events.insert_one({
                                "event_id": event_id,
                                "session_id": session_id,
                                "roll_number": roll,
                                "name": student["name"],
                                "event_type": "RETURN",
                                "timestamp": timestamp,
                                "details": "Student returned to classroom session",
                                "verified": True
                            })
                        except Exception as e:
                            print(f"[DatabaseManager] MongoDB return error: {e}")

                    return True, f"Return verified for {student['name']} ({roll}).", "RETURN"
                else:
                    # Continuous presence update
                    cursor.execute("""
                        UPDATE attendance SET
                            last_seen = ?,
                            total_duration_minutes = ?,
                            total_duration_formatted = ?
                        WHERE session_id = ? AND roll_number = ?
                    """, (timestamp, total_min, formatted_dur, session_id, roll))
                    conn.commit()

                    if self.is_mongo_connected:
                        try:
                            self.mongo_db.Attendance.update_one(
                                {"session_id": session_id, "roll_number": roll},
                                {"$set": {
                                    "last_seen": timestamp,
                                    "total_duration_minutes": total_min,
                                    "total_duration_formatted": formatted_dur
                                }}
                            )
                        except Exception:
                            pass

                    return True, f"Presence active for {student['name']}.", "ACTIVE"

    def record_student_exit(self, session_id, roll_number, reason="Classroom exit"):
        """
        Marks student as CHECKED_OUT and logs EXIT event.
        Generates neutral notification to teacher if exit count exceeds 1.
        """
        roll = str(roll_number).strip()
        student = self.get_student(roll)
        student_name = student["name"] if student else roll
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM attendance WHERE session_id = ? AND roll_number = ?", (session_id, roll))
            row = cursor.fetchone()
            if not row:
                return False, "Student does not have an active check-in record in this session."

            new_exit_count = (row["exit_count"] or 0) + 1

            cursor.execute("""
                UPDATE attendance SET
                    status = 'CHECKED_OUT',
                    exit_count = ?
                WHERE session_id = ? AND roll_number = ?
            """, (new_exit_count, session_id, roll))

            event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"
            cursor.execute("""
                INSERT INTO session_events (
                    event_id, session_id, roll_number, name,
                    event_type, timestamp, details, verified
                ) VALUES (?, ?, ?, ?, 'EXIT', ?, ?, 1)
            """, (event_id, session_id, roll, student_name, now, reason))

            # Configurable neutral notification to teacher on repeated exits (>= 2 exits)
            if new_exit_count >= 2:
                notif_id = f"NOTIF_{uuid.uuid4().hex[:8].upper()}"
                title = "Classroom Exit Pattern Review"
                msg = f"Neutral note: Student {student_name} (Roll: {roll}) recorded {new_exit_count} exit events in the current session."
                cursor.execute("""
                    INSERT INTO system_notifications (
                        notification_id, session_id, roll_number, student_name,
                        title, message, severity, timestamp, is_read
                    ) VALUES (?, ?, ?, ?, ?, ?, 'NEUTRAL_NOTICE', ?, 0)
                """, (notif_id, session_id, roll, student_name, title, msg, now))

            conn.commit()

        if self.is_mongo_connected:
            try:
                self.mongo_db.Attendance.update_one(
                    {"session_id": session_id, "roll_number": roll},
                    {"$set": {"status": "CHECKED_OUT", "exit_count": new_exit_count}}
                )
                self.mongo_db.Session_Events.insert_one({
                    "event_id": event_id,
                    "session_id": session_id,
                    "roll_number": roll,
                    "name": student_name,
                    "event_type": "EXIT",
                    "timestamp": now,
                    "details": reason,
                    "verified": True
                })
                if new_exit_count >= 2:
                    self.mongo_db.System_Notifications.insert_one({
                        "notification_id": notif_id,
                        "session_id": session_id,
                        "roll_number": roll,
                        "student_name": student_name,
                        "title": title,
                        "message": msg,
                        "severity": "NEUTRAL_NOTICE",
                        "timestamp": now,
                        "is_read": False
                    })
            except Exception as e:
                print(f"[DatabaseManager] MongoDB exit record error: {e}")

        return True, f"Exit event recorded for {student_name}."

    def manual_teacher_correction(self, session_id, roll_number, new_status, reason="Instructor correction"):
        """
        Allows teacher to review and correct any student's attendance status or duration.
        """
        roll = str(roll_number).strip()
        student = self.get_student(roll)
        student_name = student["name"] if student else roll
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            # Upsert into attendance
            cursor.execute("""
                INSERT INTO attendance (
                    attendance_id, session_id, roll_number, name,
                    department, section, status, first_check_in,
                    last_seen, total_duration_minutes, total_duration_formatted,
                    exit_count, verification_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, 'Manual Entry', 0, 'MANUAL_CORRECTION')
                ON CONFLICT(session_id, roll_number) DO UPDATE SET
                    status = excluded.status,
                    verification_method = 'MANUAL_CORRECTION'
            """, (
                f"ATT_{uuid.uuid4().hex[:8].upper()}", session_id, roll, student_name,
                student.get("department", "") if student else "",
                student.get("section", "") if student else "",
                new_status, now, now
            ))

            event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"
            cursor.execute("""
                INSERT INTO session_events (
                    event_id, session_id, roll_number, name,
                    event_type, timestamp, details, verified
                ) VALUES (?, ?, ?, ?, 'MANUAL_OVERRIDE', ?, ?, 1)
            """, (event_id, session_id, roll, student_name, now, f"{reason} -> Status set to {new_status}"))
            conn.commit()

        if self.is_mongo_connected:
            try:
                self.mongo_db.Attendance.update_one(
                    {"session_id": session_id, "roll_number": roll},
                    {"$set": {
                        "status": new_status,
                        "verification_method": "MANUAL_CORRECTION",
                        "last_seen": now
                    }},
                    upsert=True
                )
                self.mongo_db.Session_Events.insert_one({
                    "event_id": event_id,
                    "session_id": session_id,
                    "roll_number": roll,
                    "name": student_name,
                    "event_type": "MANUAL_OVERRIDE",
                    "timestamp": now,
                    "details": f"{reason} -> Status set to {new_status}",
                    "verified": True
                })
            except Exception as e:
                print(f"[DatabaseManager] MongoDB manual correction error: {e}")

        return True, f"Attendance status updated to {new_status} for {student_name}."

    def get_session_attendance(self, session_id):
        """Retrieves attendance records for a specific session."""
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT a.*, s.profile_photo
                FROM attendance a
                LEFT JOIN students s ON a.roll_number = s.roll_number
                WHERE a.session_id = ?
                ORDER BY a.first_check_in DESC
            """, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_session_roster_status(self, session_id, section=None):
        """
        Returns full section roster with present/absent status for active session.
        """
        students = self.get_all_students(section=section)
        session_records = {r["roll_number"]: r for r in self.get_session_attendance(session_id)}

        roster = []
        for s in students:
            roll = s["roll_number"]
            rec = session_records.get(roll)
            if rec:
                roster.append({
                    "roll_number": roll,
                    "name": s["name"],
                    "department": s.get("department", "Engineering"),
                    "section": s.get("section", "A"),
                    "profile_photo": s.get("profile_photo") or f"/static/profiles/{roll}.jpg",
                    "status": rec["status"],
                    "first_check_in": rec["first_check_in"],
                    "last_seen": rec["last_seen"],
                    "duration_formatted": rec["total_duration_formatted"],
                    "exit_count": rec["exit_count"],
                    "verification_method": rec["verification_method"],
                    "is_present": rec["status"] in ["PRESENT", "CHECKED_OUT"]
                })
            else:
                roster.append({
                    "roll_number": roll,
                    "name": s["name"],
                    "department": s.get("department", "Engineering"),
                    "section": s.get("section", "A"),
                    "profile_photo": s.get("profile_photo") or f"/static/profiles/{roll}.jpg",
                    "status": "ABSENT",
                    "first_check_in": None,
                    "last_seen": None,
                    "duration_formatted": "0m 0s",
                    "exit_count": 0,
                    "verification_method": None,
                    "is_present": False
                })
        return roster

    def get_session_events(self, session_id, limit=30):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM session_events
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_session_notifications(self, session_id, unread_only=False):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if unread_only:
                cursor.execute("SELECT * FROM system_notifications WHERE session_id = ? AND is_read = 0 ORDER BY timestamp DESC", (session_id,))
            else:
                cursor.execute("SELECT * FROM system_notifications WHERE session_id = ? ORDER BY timestamp DESC", (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def mark_notification_read(self, notification_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("UPDATE system_notifications SET is_read = 1 WHERE notification_id = ?", (notification_id,))
            conn.commit()
        if self.is_mongo_connected:
            try:
                self.mongo_db.System_Notifications.update_one({"notification_id": notification_id}, {"$set": {"is_read": True}})
            except Exception:
                pass
        return True

    def export_session_csv(self, session_id):
        """Exports session attendance records to a downloadable CSV file."""
        roster = self.get_session_roster_status(session_id)
        df = pd.DataFrame(roster)
        if df.empty:
            df = pd.DataFrame(columns=[
                "roll_number", "name", "department", "section", "status",
                "first_check_in", "last_seen", "duration_formatted", "exit_count", "verification_method"
            ])

        df.rename(columns={
            "roll_number": "Roll Number",
            "name": "Student Name",
            "department": "Department",
            "section": "Section",
            "status": "Attendance Status",
            "first_check_in": "First Check-In Time",
            "last_seen": "Last Seen Time",
            "duration_formatted": "Total Verified Presence",
            "exit_count": "Exit Count",
            "verification_method": "Verification Source"
        }, inplace=True)

        target_dir = BASE_DIR / "database" / "exports"
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"Session_{session_id}_Attendance.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    # =========================================================================
    # 4. LEGACY BACKWARD COMPATIBILITY
    # =========================================================================
    def sync_from_csv(self):
        if not CSV_ENROLLMENT_PATH.exists():
            return
        try:
            df = pd.read_csv(CSV_ENROLLMENT_PATH)
            for _, row in df.iterrows():
                roll = str(row.get("Roll Number", "")).strip()
                name = str(row.get("Name", "")).strip()
                dept = str(row.get("Department", "Textile Technology")).strip()
                if not dept or dept == "nan":
                    dept = "Textile Technology"
                sec = str(row.get("Section", "Section A")).strip()
                if not sec or sec == "nan":
                    sec = "Section A"
                consent = row.get("Consent Given", 1)
                try:
                    consent_val = int(consent) if pd.notna(consent) else 1
                except Exception:
                    consent_val = 1
                photo = str(row.get("Profile Photo", f"/static/profiles/{roll}.jpg")).strip()
                if roll and name and roll != "nan" and name != "nan":
                    self.enroll_student(roll, name, department=dept, section=sec, profile_photo=photo, consent_given=consent_val, sync_csv=False)
        except Exception as e:
            print(f"[DatabaseManager] Warning reading CSV sync: {e}")

    def _update_enrollment_csv(self, roll, name, department="Textile Technology", section="Section A", profile_photo="", consent_given=1):
        try:
            cols = ["Roll Number", "Name", "Department", "Section", "Profile Photo", "Consent Given"]
            if CSV_ENROLLMENT_PATH.exists():
                try:
                    df = pd.read_csv(CSV_ENROLLMENT_PATH)
                except Exception:
                    df = pd.DataFrame(columns=cols)
            else:
                df = pd.DataFrame(columns=cols)

            for col in cols:
                if col not in df.columns:
                    df[col] = ""

            df["Roll Number"] = df["Roll Number"].astype(str)
            photo_path = profile_photo or f"/static/profiles/{roll}.jpg"
            consent_int = 1 if consent_given else 0

            if roll in df["Roll Number"].values:
                df.loc[df["Roll Number"] == roll, ["Name", "Department", "Section", "Profile Photo", "Consent Given"]] = [
                    name, department, section, photo_path, consent_int
                ]
            else:
                new_row = pd.DataFrame([{
                    "Roll Number": roll,
                    "Name": name,
                    "Department": department,
                    "Section": section,
                    "Profile Photo": photo_path,
                    "Consent Given": consent_int
                }])
                df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(CSV_ENROLLMENT_PATH, index=False)
        except Exception as e:
            print(f"[DatabaseManager] CSV update error: {e}")

    def mark_attendance(self, subject, roll_number, name=None):
        """Legacy helper for Hindi / English routes."""
        active = self.get_active_session()
        sess_id = active["session_id"] if active else "LEGACY_SESSION"
        success, msg, event_type = self.record_session_presence(sess_id, roll_number)

        # Also update legacy tables
        subj = subject.strip().capitalize()
        table = "hindi_attendance" if subj == "Hindi" else "english_attendance"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {table} SET attendance_count = attendance_count + 1, last_marked = ? WHERE roll_number = ?", (now, str(roll_number)))
            conn.commit()

        return success, msg, 1

    def get_attendance(self, subject):
        subj = subject.strip().capitalize()
        table = "hindi_attendance" if subj == "Hindi" else "english_attendance"
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"SELECT roll_number, name, attendance_count, last_marked FROM {table} ORDER BY roll_number ASC")
            return [dict(row) for row in cursor.fetchall()]

    def manual_update_attendance(self, subject, roll_number, new_count):
        roll = str(roll_number).strip()
        subj = subject.strip().capitalize()
        table = "hindi_attendance" if subj == "Hindi" else "english_attendance"
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE {table} SET attendance_count = ? WHERE roll_number = ?", (int(new_count), roll))
            conn.commit()
        return True, f"Updated {subj} attendance for Roll {roll} to {new_count}."

    def get_recent_logs(self, limit=25):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, subject, roll_number, name, timestamp, date FROM attendance_logs ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def export_csv(self, subject):
        subj = subject.strip().capitalize()
        records = self.get_attendance(subj)
        df = pd.DataFrame(records)
        target_dir = BASE_DIR / f"{subj}_attendance"
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_file = target_dir / f"{subj}_Attendance.csv"
        df.to_csv(csv_file, index=False)
        return str(csv_file)

    # =========================================================================
    # 4. STUDENT & FACULTY FACE EMBEDDINGS STORAGE
    # =========================================================================
    def save_student_embedding(self, student_id, embedding, quality_score=0.0, blur_score=0.0, is_valid=1, model_version="FaceNet-128D"):
        """Stores a 128-D normalized embedding vector for an enrolled student."""
        emb_id = f"EMB_{uuid.uuid4().hex[:10].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_list = embedding if isinstance(embedding, list) else embedding.tolist()
        emb_json = json.dumps(emb_list)

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.cursor().execute("""
                    INSERT INTO student_face_embeddings (
                        embedding_id, student_id, embedding, model_version,
                        quality_score, blur_score, is_valid, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (emb_id, str(student_id), emb_json, model_version, float(quality_score), float(blur_score), int(is_valid), now))
                conn.commit()
        except Exception as e:
            print(f"[DatabaseManager] save_student_embedding SQLite error: {e}")

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Student_Face_Embeddings.insert_one({
                    "embedding_id": emb_id,
                    "student_id": str(student_id),
                    "embedding": emb_list,
                    "model_version": model_version,
                    "quality_score": float(quality_score),
                    "blur_score": float(blur_score),
                    "is_valid": int(is_valid),
                    "created_at": now
                })
            except Exception as e:
                print(f"[DatabaseManager] save_student_embedding Mongo error: {e}")
        return emb_id

    def get_student_embeddings(self, student_id):
        """Retrieves all stored 128-D embedding vectors for a student."""
        sid = str(student_id).strip()
        vectors = []
        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                for doc in self.mongo_db.Student_Face_Embeddings.find({"student_id": sid}):
                    vectors.append(doc.get("embedding"))
                if vectors:
                    return vectors
            except Exception:
                pass

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.cursor().execute("SELECT embedding FROM student_face_embeddings WHERE student_id = ?", (sid,)).fetchall():
                    vectors.append(json.loads(r["embedding"]))
        except Exception:
            pass
        return vectors

    # =========================================================================
    # 5. FACULTY MANAGEMENT & FACULTY FACE EMBEDDINGS (Strict Exclusion)
    # =========================================================================
    def enroll_faculty(self, faculty_id, name, department, designation="Assistant Professor",
                       email="", phone="", profile_photo=None, face_enrollment_status="PENDING"):
        """Enrolls a faculty member into isolated faculty storage."""
        fac_id = str(faculty_id).strip()
        fac_name = str(name).strip()
        if not fac_id or not fac_name:
            return False, "Faculty ID and name cannot be empty."

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photo_url = profile_photo or f"/static/profiles/faculty_{fac_id}.jpg"

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO faculty (
                    faculty_id, name, email, department, designation,
                    phone, profile_photo, face_enrollment_status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(faculty_id) DO UPDATE SET
                    name=excluded.name, email=excluded.email, department=excluded.department,
                    designation=excluded.designation, phone=excluded.phone,
                    profile_photo=excluded.profile_photo,
                    face_enrollment_status=COALESCE(excluded.face_enrollment_status, faculty.face_enrollment_status)
            """, (fac_id, fac_name, email, department, designation, phone, photo_url, face_enrollment_status, now))
            # Also keep teachers table synced for backward compatibility
            cursor.execute("""
                INSERT OR REPLACE INTO teachers (teacher_id, name, department, email, phone, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (fac_id, fac_name, department, email, phone, now))
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Faculty.update_one(
                    {"faculty_id": fac_id},
                    {"$set": {
                        "name": fac_name, "email": email, "department": department,
                        "designation": designation, "phone": phone, "profile_photo": photo_url,
                        "face_enrollment_status": face_enrollment_status, "updated_at": now
                    }, "$setOnInsert": {"created_at": now}},
                    upsert=True
                )
                self.mongo_db.Teachers.update_one(
                    {"teacher_id": fac_id},
                    {"$set": {"name": fac_name, "department": department, "email": email, "phone": phone, "created_at": now}},
                    upsert=True
                )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB enroll_faculty error: {e}")

        self._get_faculty_folder(fac_id, create=True)
        return True, f"Faculty member {fac_name} ({fac_id}) enrolled successfully."

    def get_all_faculty(self):
        """Returns all faculty members with photo counts and enrollment statuses."""
        faculty_map = {}
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.cursor().execute("SELECT * FROM faculty ORDER BY name").fetchall():
                    d = dict(r)
                    faculty_map[d["faculty_id"]] = d
        except Exception:
            pass

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                for doc in self.mongo_db.Faculty.find():
                    doc.pop("_id", None)
                    fid = doc.get("faculty_id")
                    if fid:
                        if fid not in faculty_map:
                            faculty_map[fid] = doc
                        else:
                            faculty_map[fid].update(doc)
            except Exception:
                pass

        # Also merge any teachers not yet in faculty table
        for t in self.get_all_teachers():
            tid = t.get("teacher_id")
            if tid and tid not in faculty_map:
                faculty_map[tid] = {
                    "faculty_id": tid,
                    "name": t.get("name"),
                    "department": t.get("department", "Engineering"),
                    "designation": "Faculty Member",
                    "email": t.get("email", ""),
                    "phone": t.get("phone", ""),
                    "profile_photo": f"/static/profiles/faculty_{tid}.jpg",
                    "face_enrollment_status": "PENDING",
                    "created_at": t.get("created_at", "")
                }

        # Attach photo counts
        for fid, f_data in faculty_map.items():
            photos = self.get_faculty_photos(fid)
            f_data["photo_count"] = len(photos)
            if len(photos) > 0 and f_data.get("face_enrollment_status") == "PENDING":
                f_data["face_enrollment_status"] = "ENROLLED"

        return sorted(list(faculty_map.values()), key=lambda x: str(x.get("name", "")))

    def get_faculty(self, faculty_id):
        fid = str(faculty_id).strip()
        fac = None
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                r = conn.cursor().execute("SELECT * FROM faculty WHERE faculty_id = ?", (fid,)).fetchone()
                if r:
                    fac = dict(r)
        except Exception:
            pass

        if not fac and self.is_mongo_connected and self.mongo_db is not None:
            try:
                doc = self.mongo_db.Faculty.find_one({"faculty_id": fid})
                if doc:
                    doc.pop("_id", None)
                    fac = doc
            except Exception:
                pass

        if fac:
            photos = self.get_faculty_photos(fid)
            fac["photos"] = photos
            fac["photo_count"] = len(photos)
        return fac

    def delete_faculty(self, faculty_id):
        fid = str(faculty_id).strip()
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.cursor().execute("DELETE FROM faculty WHERE faculty_id = ?", (fid,))
            conn.cursor().execute("DELETE FROM faculty_face_embeddings WHERE faculty_id = ?", (fid,))
            conn.cursor().execute("DELETE FROM teachers WHERE teacher_id = ?", (fid,))
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Faculty.delete_one({"faculty_id": fid})
                self.mongo_db.Faculty_Face_Embeddings.delete_many({"faculty_id": fid})
                self.mongo_db.Teachers.delete_one({"teacher_id": fid})
            except Exception:
                pass

        # Purge faculty photo folder
        folder = self._get_faculty_folder(fid)
        if folder and folder.exists():
            shutil.rmtree(str(folder), ignore_errors=True)

        # Remove from faculty matcher in memory
        try:
            from ai_engine import faculty_matcher
            if fid in faculty_matcher.prototypes:
                del faculty_matcher.prototypes[fid]
                faculty_matcher.save_prototypes()
        except Exception:
            pass

        return True, f"Faculty member {fid} removed."

    def _get_faculty_folder(self, faculty_id, create=False):
        fid = str(faculty_id).strip()
        faculty_dir = BASE_DIR / "faculty_people"
        faculty_dir.mkdir(parents=True, exist_ok=True)

        for folder in faculty_dir.iterdir():
            if folder.is_dir() and (folder.name.startswith(f"{fid}_") or folder.name == fid):
                return folder

        if create:
            name = fid
            try:
                with sqlite3.connect(SQLITE_PATH) as conn:
                    row = conn.cursor().execute("SELECT name FROM faculty WHERE faculty_id = ?", (fid,)).fetchone()
                    if row and row[0]:
                        name = row[0]
            except Exception:
                pass
            clean_name = re.sub(r"[^\w\-_]", "", name.replace(" ", "_"))
            new_folder = faculty_dir / f"{fid}_{clean_name}"
            new_folder.mkdir(parents=True, exist_ok=True)
            return new_folder
        return None

    def get_faculty_photos(self, faculty_id):
        folder = self._get_faculty_folder(faculty_id)
        if not folder or not folder.exists():
            return []

        photos = []
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and f.is_file():
                photos.append({
                    "filename": f.name,
                    "url": f"/faculty_photos/{folder.name}/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "modified": datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
        return photos[:10]

    def add_faculty_photo(self, faculty_id, image_bytes, filename=None):
        fid = str(faculty_id).strip()
        fac = self.get_faculty(fid)
        if not fac:
            return False, "Faculty member not found."

        folder = self._get_faculty_folder(fid, create=True)
        existing = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(existing) >= 10:
            return False, "Maximum 10 photographs reached for this faculty member."

        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False, "Invalid image format. Supported formats: JPEG, PNG."

        used_numbers = []
        for f in existing:
            m = re.match(r"^(\d+)\.", f.name)
            if m:
                used_numbers.append(int(m.group(1)))

        next_num = 1
        for n in range(1, 11):
            if n not in used_numbers:
                next_num = n
                break

        dest_file = folder / f"{next_num}.jpg"
        cv2.imwrite(str(dest_file), img)

        prof_file = PROFILES_DIR / f"faculty_{fid}.jpg"
        if not prof_file.exists() or len(existing) == 0:
            cv2.imwrite(str(prof_file), img)

        # Trigger immediate automatic face enrollment for faculty
        enroll_notice = ""
        try:
            from ai_engine import auto_enroll_faculty_face
            enroll_res = auto_enroll_faculty_face(fid, fac.get("name", fid), img)
            if enroll_res.get("success"):
                enroll_notice = " Biometric face model enrolled and activated in Faculty Exclusion namespace."
            else:
                enroll_notice = f" Enrollment status: {enroll_res.get('message', 'Processing issue.')}"
        except Exception as e:
            enroll_notice = f" Enrollment notice: {e}"

        return True, f"Faculty photograph saved as {dest_file.name}.{enroll_notice}"

    def delete_faculty_photo(self, faculty_id, filename):
        folder = self._get_faculty_folder(faculty_id)
        if not folder:
            return False, "Faculty folder not found."
        target = folder / filename
        if target.exists():
            target.unlink()
            remaining = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
            if remaining:
                try:
                    from ai_engine import auto_enroll_faculty_face
                    auto_enroll_faculty_face(faculty_id, "Faculty", cv2.imread(str(remaining[0])))
                except Exception:
                    pass
            return True, f"Photo {filename} deleted."
        return False, "Photo not found."

    def save_faculty_embedding(self, faculty_id, name, embedding, model_version="FaceNet-128D"):
        emb_id = f"FAC_EMB_{uuid.uuid4().hex[:10].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_list = embedding if isinstance(embedding, list) else embedding.tolist()
        emb_json = json.dumps(emb_list)

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.cursor().execute("""
                    INSERT INTO faculty_face_embeddings (
                        embedding_id, faculty_id, name, embedding, model_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (emb_id, str(faculty_id), name, emb_json, model_version, now))
                conn.commit()
        except Exception as e:
            print(f"[DatabaseManager] save_faculty_embedding SQLite error: {e}")

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Faculty_Face_Embeddings.insert_one({
                    "embedding_id": emb_id,
                    "faculty_id": str(faculty_id),
                    "name": name,
                    "embedding": emb_list,
                    "model_version": model_version,
                    "created_at": now
                })
            except Exception as e:
                print(f"[DatabaseManager] save_faculty_embedding Mongo error: {e}")
        return emb_id

    def get_faculty_embeddings(self, faculty_id):
        fid = str(faculty_id).strip()
        vectors = []
        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                for doc in self.mongo_db.Faculty_Face_Embeddings.find({"faculty_id": fid}):
                    vectors.append(doc.get("embedding"))
                if vectors:
                    return vectors
            except Exception:
                pass

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.cursor().execute("SELECT embedding FROM faculty_face_embeddings WHERE faculty_id = ?", (fid,)).fetchall():
                    vectors.append(json.loads(r["embedding"]))
        except Exception:
            pass
        return vectors

    # =========================================================================
    # 6. RECOGNITION AUDIT LOGGING & ZERO-TOLERANCE ANALYTICS
    # =========================================================================
    def log_recognition_decision(self, session_id, predicted_identity_type,
                                 matched_identity_id, matched_name, similarity_score,
                                 decision, rejection_reason="", model_version="FaceNet-128D"):
        """Logs every camera recognition decision for real-time auditability."""
        log_id = f"LOG_{uuid.uuid4().hex[:10].upper()}"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.cursor().execute("""
                    INSERT INTO recognition_logs (
                        log_id, session_id, predicted_identity_type, matched_identity_id,
                        matched_name, similarity_score, decision, rejection_reason,
                        model_version, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id, session_id or "", predicted_identity_type, matched_identity_id or "",
                    matched_name or "", float(similarity_score or 0.0), decision,
                    rejection_reason or "", model_version, now
                ))
                conn.commit()
        except Exception:
            pass

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Recognition_Logs.insert_one({
                    "log_id": log_id,
                    "session_id": session_id or "",
                    "predicted_identity_type": predicted_identity_type,
                    "matched_identity_id": matched_identity_id or "",
                    "matched_name": matched_name or "",
                    "similarity_score": float(similarity_score or 0.0),
                    "decision": decision,
                    "rejection_reason": rejection_reason or "",
                    "model_version": model_version,
                    "timestamp": now
                })
            except Exception:
                pass
        return log_id

    def get_recognition_logs(self, session_id=None, limit=50):
        """Retrieves recent camera recognition decisions and audit records."""
        logs = []
        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                q = {"session_id": session_id} if session_id else {}
                cursor = self.mongo_db.Recognition_Logs.find(q).sort("timestamp", -1).limit(limit)
                for doc in cursor:
                    doc.pop("_id", None)
                    logs.append(doc)
                if logs:
                    return logs
            except Exception:
                pass

        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if session_id:
                    cursor.execute("SELECT * FROM recognition_logs WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit))
                else:
                    cursor.execute("SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
                return [dict(r) for r in cursor.fetchall()]
        except Exception:
            return []

    def get_all_class_groups(self):
        """Aggregates all unique registered class groups with their active roster counts."""
        students = self.get_all_students()
        groups_dict = {}
        for s in students:
            deg = s.get("degree", "B.Tech")
            strm = s.get("stream", s.get("department", "Computer Science & Engineering"))
            yr = s.get("academic_year", "1st Year")
            sess = s.get("academic_session", "2026-27")
            key = f"{deg}___{strm}___{yr}___{sess}"
            if key not in groups_dict:
                groups_dict[key] = {
                    "degree": deg,
                    "stream": strm,
                    "academic_year": yr,
                    "academic_session": sess,
                    "student_count": 0,
                    "enrolled_faces": 0
                }
            groups_dict[key]["student_count"] += 1
            if s.get("face_enrollment_status") == "ENROLLED":
                groups_dict[key]["enrolled_faces"] += 1

        return sorted(list(groups_dict.values()), key=lambda x: (x["degree"], x["stream"], x["academic_year"]))

    def get_dashboard_analytics(self):
        """Calculates authentic, real-time database-backed analytics without fabrication."""
        students = self.get_all_students()
        faculty = self.get_all_faculty()
        sessions = self.get_all_sessions(limit=100)
        logs = self.get_recognition_logs(limit=250)

        total_students = len(students)
        total_faculty = len(faculty)
        enrolled_faces = len([s for s in students if s.get("face_enrollment_status") == "ENROLLED"])
        failed_enrollments = len([s for s in students if str(s.get("face_enrollment_status", "")).startswith("FAILED")])
        active_sessions = len([s for s in sessions if s.get("status") == "ACTIVE"])

        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        today_attendance = 0
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM attendance WHERE first_check_in LIKE ?", (f"{today_date}%",))
                today_attendance = c.fetchone()[0]
        except Exception:
            pass

        outside_class_count = len([l for l in logs if l.get("predicted_identity_type") == "OUTSIDE_CLASS"])
        unknown_count = len([l for l in logs if l.get("predicted_identity_type") == "UNKNOWN"])
        rejected_count = len([l for l in logs if "REJECTED" in str(l.get("decision", ""))])

        degree_dist = {}
        stream_dist = {}
        year_dist = {}
        class_groups = set()
        for s in students:
            deg = s.get("degree", "B.Tech")
            strm = s.get("stream", s.get("department", "CSE"))
            yr = s.get("academic_year", "1st Year")
            sess = s.get("academic_session", "2026-27")
            degree_dist[deg] = degree_dist.get(deg, 0) + 1
            stream_dist[strm] = stream_dist.get(strm, 0) + 1
            year_dist[yr] = year_dist.get(yr, 0) + 1
            class_groups.add(f"{deg}_{strm}_{yr}_{sess}")

        return {
            "total_students": total_students,
            "total_faculty": total_faculty,
            "enrolled_faces": enrolled_faces,
            "failed_enrollments": failed_enrollments,
            "total_class_groups": len(class_groups),
            "active_sessions": active_sessions,
            "today_attendance": today_attendance,
            "outside_class_count": outside_class_count,
            "unknown_count": unknown_count,
            "rejected_count": rejected_count,
            "degree_distribution": degree_dist,
            "stream_distribution": stream_dist,
            "year_distribution": year_dist,
            "recent_logs": logs[:15]
        }

    def migrate_schema_and_data(self):
        """Automatically populates missing schema columns and normalizes student profiles."""
        try:
            with sqlite3.connect(SQLITE_PATH) as conn:
                cursor = conn.cursor()
                # Populate default academic info for any legacy student records
                cursor.execute("""
                    UPDATE students SET
                        degree = COALESCE(NULLIF(degree, ''), 'B.Tech'),
                        stream = COALESCE(NULLIF(stream, ''), department, 'Computer Science & Engineering'),
                        academic_year = COALESCE(NULLIF(academic_year, ''), '1st Year'),
                        academic_session = COALESCE(NULLIF(academic_session, ''), '2026-27'),
                        face_enrollment_status = COALESCE(NULLIF(face_enrollment_status, ''), 'ENROLLED')
                    WHERE degree IS NULL OR stream IS NULL OR academic_year IS NULL OR academic_session IS NULL
                """)
                conn.commit()
        except Exception as e:
            print(f"[DatabaseManager] Migration note: {e}")

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Students.update_many(
                    {"degree": {"$exists": False}},
                    {"$set": {"degree": "B.Tech", "academic_year": "1st Year", "academic_session": "2026-27", "face_enrollment_status": "ENROLLED"}}
                )
            except Exception:
                pass



    # =========================================================================
    # MODULE 1: STUDENT MANAGEMENT & BULK IMPORT/EXPORT
    # =========================================================================

    def add_student_record(self, name, roll_number, student_id=None, class_name=None,
                           section=None, academic_session=None, profile_photo=None,
                           enrollment_status="Active", department="Computer Science & Engineering",
                           degree="B.Tech", stream=None, academic_year="1st Year",
                           email="", mobile=""):
        """Creates or registers a student record with all Module 1 required fields."""
        roll = str(roll_number).strip()
        st_id = (student_id or roll).strip()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stream_val = stream or department

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO students (
                    roll_number, student_id, name, class_name, section, academic_session,
                    profile_photo, enrollment_status, department, degree, stream,
                    academic_year, email, mobile, consent_given, consent_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(roll_number) DO UPDATE SET
                    student_id = excluded.student_id,
                    name = excluded.name,
                    class_name = excluded.class_name,
                    section = excluded.section,
                    academic_session = excluded.academic_session,
                    profile_photo = COALESCE(excluded.profile_photo, students.profile_photo),
                    enrollment_status = excluded.enrollment_status,
                    department = excluded.department,
                    degree = excluded.degree,
                    stream = excluded.stream,
                    academic_year = excluded.academic_year,
                    email = excluded.email,
                    mobile = excluded.mobile
            """, (
                roll, st_id, name.strip(), class_name or f"{degree} {academic_year}",
                section or "Section A", academic_session or "2026-27", profile_photo or "",
                enrollment_status, department, degree, stream_val, academic_year,
                email, mobile, now_str, now_str
            ))
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Students.update_one(
                    {"roll_number": roll},
                    {"$set": {
                        "roll_number": roll,
                        "student_id": st_id,
                        "name": name.strip(),
                        "class_name": class_name or f"{degree} {academic_year}",
                        "section": section or "Section A",
                        "academic_session": academic_session or "2026-27",
                        "profile_photo": profile_photo or "",
                        "enrollment_status": enrollment_status,
                        "department": department,
                        "degree": degree,
                        "stream": stream_val,
                        "academic_year": academic_year,
                        "email": email,
                        "mobile": mobile,
                        "created_at": now_str
                    }},
                    upsert=True
                )
            except Exception as e:
                print(f"[DatabaseManager] MongoDB student sync error: {e}")

        return self.get_student(roll)

    def edit_student_record(self, roll_number, name=None, student_id=None, class_name=None,
                            section=None, academic_session=None, enrollment_status=None,
                            department=None, degree=None, stream=None, academic_year=None,
                            email=None, mobile=None):
        """Updates an existing student profile with full audit-trail safety."""
        roll = str(roll_number).strip()
        curr = self.get_student(roll)
        if not curr:
            return None

        new_name = name.strip() if name is not None else curr.get("name", "")
        new_sid = student_id.strip() if student_id is not None else curr.get("student_id", roll)
        new_class = class_name if class_name is not None else curr.get("class_name", "")
        new_sec = section if section is not None else curr.get("section", "")
        new_sess = academic_session if academic_session is not None else curr.get("academic_session", "2026-27")
        new_status = enrollment_status if enrollment_status is not None else curr.get("enrollment_status", "Active")
        new_dept = department if department is not None else curr.get("department", "Computer Science & Engineering")
        new_deg = degree if degree is not None else curr.get("degree", "B.Tech")
        new_stream = stream if stream is not None else curr.get("stream", new_dept)
        new_yr = academic_year if academic_year is not None else curr.get("academic_year", "1st Year")
        new_email = email if email is not None else curr.get("email", "")
        new_mob = mobile if mobile is not None else curr.get("mobile", "")

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE students SET
                    name = ?, student_id = ?, class_name = ?, section = ?,
                    academic_session = ?, enrollment_status = ?, department = ?,
                    degree = ?, stream = ?, academic_year = ?, email = ?, mobile = ?
                WHERE roll_number = ?
            """, (
                new_name, new_sid, new_class, new_sec, new_sess, new_status,
                new_dept, new_deg, new_stream, new_yr, new_email, new_mob, roll
            ))
            conn.commit()

        if self.is_mongo_connected and self.mongo_db is not None:
            try:
                self.mongo_db.Students.update_one(
                    {"roll_number": roll},
                    {"$set": {
                        "name": new_name, "student_id": new_sid, "class_name": new_class,
                        "section": new_sec, "academic_session": new_sess, "enrollment_status": new_status,
                        "department": new_dept, "degree": new_deg, "stream": new_stream,
                        "academic_year": new_yr, "email": new_email, "mobile": new_mob
                    }}
                )
            except Exception:
                pass

        return self.get_student(roll)

    def search_filter_students(self, query="", class_name="", section="", status="", stream=""):
        """Filters students by search keyword, class, section, and enrollment status."""
        all_students = self.get_all_students()
        q = (query or "").strip().lower()
        cls_filter = (class_name or "").strip().lower()
        sec_filter = (section or "").strip().lower()
        st_filter = (status or "").strip().lower()
        stream_filter = (stream or "").strip().lower()

        filtered = []
        for s in all_students:
            s_name = str(s.get("name", "")).lower()
            s_roll = str(s.get("roll_number", "")).lower()
            s_sid = str(s.get("student_id", "")).lower()
            s_cls = str(s.get("class_name", "") or f"{s.get('degree', '')} {s.get('academic_year', '')}").lower()
            s_sec = str(s.get("section", "")).lower()
            s_st = str(s.get("enrollment_status", "")).lower()

            if q and not (q in s_name or q in s_roll or q in s_sid):
                continue
            if cls_filter and cls_filter not in s_cls:
                continue
            if sec_filter and sec_filter != s_sec:
                continue
            if st_filter and st_filter != s_st:
                continue
            s_stream = str(s.get("stream", "") or s.get("department", "")).lower()
            if stream_filter and stream_filter not in s_stream:
                continue
            filtered.append(s)

        return filtered

    def bulk_import_students(self, records):
        """
        Imports student records from a list of dictionaries (from CSV/Excel).
        Validates columns and prevents duplicate errors.
        """
        imported_count = 0
        skipped_count = 0
        errors = []

        for idx, row in enumerate(records, start=1):
            try:
                # Handle flexible header naming
                name = (row.get("name") or row.get("full_name") or row.get("student_name") or row.get("Student Name") or "").strip()
                roll = (row.get("roll_number") or row.get("roll") or row.get("roll_no") or row.get("Roll Number") or "").strip()
                st_id = (row.get("student_id") or row.get("id") or row.get("Student ID") or roll).strip()
                cls = (row.get("class") or row.get("class_name") or row.get("Class") or "B.Tech 1st Year").strip()
                sec = (row.get("section") or row.get("Section") or "Section A").strip()
                sess = (row.get("academic_session") or row.get("session") or row.get("Session") or "2026-27").strip()
                status = (row.get("status") or row.get("enrollment_status") or row.get("Status") or "Active").strip()
                degree = (row.get("degree") or row.get("Degree") or "B.Tech").strip()
                stream = (row.get("stream") or row.get("branch") or row.get("Branch") or row.get("department") or "Textile Technology").strip()
                email = (row.get("email") or row.get("Email") or "").strip()
                mobile = (row.get("mobile") or row.get("phone") or row.get("Mobile") or "").strip()

                if not name or not roll:
                    skipped_count += 1
                    errors.append(f"Row {idx}: Missing Student Name or Roll Number.")
                    continue

                self.add_student_record(
                    name=name,
                    roll_number=roll,
                    student_id=st_id,
                    class_name=cls,
                    section=sec,
                    academic_session=sess,
                    enrollment_status=status,
                    degree=degree,
                    stream=stream,
                    department=stream,
                    email=email,
                    mobile=mobile
                )
                imported_count += 1
            except Exception as e:
                skipped_count += 1
                errors.append(f"Row {idx} ({row.get('roll_number', 'unknown')}): {e}")

        return {
            "success": True,
            "total_rows": len(records),
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "errors": errors
        }

    # =========================================================================
    # MODULE 2: IMAGE DATASET MANAGEMENT & STORAGE
    # =========================================================================

    def create_dataset(self, name, description="", task_type="person_detection",
                       train_split=0.70, val_split=0.15, test_split=0.15):
        ds_id = f"DS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex().upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (
                    id, name, description, task_type, total_images, total_size_bytes,
                    train_split, val_split, test_split, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, 'Ready', ?, ?)
            """, (ds_id, name.strip(), description.strip(), task_type, train_split, val_split, test_split, now_str, now_str))
            conn.commit()

        # Create physical directory
        ds_dir = BASE_DIR / "static" / "datasets" / ds_id
        ds_dir.mkdir(parents=True, exist_ok=True)
        return self.get_dataset(ds_id)

    def get_all_datasets(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
            return [dict(r) for r in cursor.fetchall()]

    def get_dataset(self, dataset_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_dataset(self, dataset_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM dataset_files WHERE dataset_id = ?", (dataset_id,))
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()

        ds_dir = BASE_DIR / "static" / "datasets" / dataset_id
        if ds_dir.exists():
            import shutil
            shutil.rmtree(ds_dir, ignore_errors=True)
        return True

    def check_dataset_duplicate(self, dataset_id, file_hash):
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM dataset_files WHERE dataset_id = ? AND file_hash = ?", (dataset_id, file_hash))
            return cursor.fetchone()[0] > 0

    def add_dataset_file(self, dataset_id, file_path, file_name, file_hash, file_size,
                         width=0, height=0, quality_score=100.0, is_corrupted=0,
                         is_duplicate=0, is_low_light=0, is_blurred=0, split_category="train"):
        f_id = f"FILE_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(3).hex().upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dataset_files (
                    id, dataset_id, file_path, file_name, file_hash, file_size,
                    width, height, split_category, quality_score, is_corrupted,
                    is_duplicate, is_low_light, is_blurred, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f_id, dataset_id, file_path, file_name, file_hash, file_size,
                width, height, split_category, quality_score, is_corrupted,
                is_duplicate, is_low_light, is_blurred, now_str
            ))
            # Update dataset summary totals
            cursor.execute("""
                UPDATE datasets SET
                    total_images = (SELECT COUNT(*) FROM dataset_files WHERE dataset_id = ?),
                    total_size_bytes = (SELECT COALESCE(SUM(file_size), 0) FROM dataset_files WHERE dataset_id = ?),
                    updated_at = ?
                WHERE id = ?
            """, (dataset_id, dataset_id, now_str, dataset_id))
            conn.commit()

        return f_id

    def get_dataset_files(self, dataset_id, split=None):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if split:
                cursor.execute("SELECT * FROM dataset_files WHERE dataset_id = ? AND split_category = ? ORDER BY created_at DESC", (dataset_id, split))
            else:
                cursor.execute("SELECT * FROM dataset_files WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,))
            return [dict(r) for r in cursor.fetchall()]

    def delete_dataset_file(self, file_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dataset_files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if not row:
                return False
            rec = dict(row)
            cursor.execute("DELETE FROM dataset_files WHERE id = ?", (file_id,))
            # Update dataset summary
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
                UPDATE datasets SET
                    total_images = (SELECT COUNT(*) FROM dataset_files WHERE dataset_id = ?),
                    total_size_bytes = (SELECT COALESCE(SUM(file_size), 0) FROM dataset_files WHERE dataset_id = ?),
                    updated_at = ?
                WHERE id = ?
            """, (rec["dataset_id"], rec["dataset_id"], now_str, rec["dataset_id"]))
            conn.commit()

        # Remove physical file
        try:
            p = BASE_DIR / rec["file_path"].lstrip("/").replace("/", os.sep)
            if p.exists():
                p.unlink()
        except Exception:
            pass
        return True

    # =========================================================================
    # MODULE 3 & 4: AI TRAINING STUDIO & ASYNCHRONOUS MONITOR
    # =========================================================================

    def create_training_job(self, job_name, dataset_id, task_type,
                            model_architecture="MobileNet-PersonDetector",
                            epochs=10, batch_size=16, learning_rate=0.001,
                            optimizer="adam"):
        j_id = f"JOB_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex().upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_jobs (
                    id, job_name, dataset_id, task_type, model_architecture, status,
                    total_epochs, current_epoch, batch_size, learning_rate, optimizer,
                    train_loss, val_loss, train_acc, val_acc, precision, recall, mAP,
                    progress_percent, model_version, error_message, metrics_history,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'v1.0', '', '[]', ?)
            """, (
                j_id, job_name, dataset_id, task_type, model_architecture,
                epochs, batch_size, learning_rate, optimizer, now_str
            ))
            conn.commit()

        return self.get_training_job(j_id)

    def update_training_job_progress(self, job_id, current_epoch, train_loss, val_loss,
                                     train_acc, val_acc, precision, recall, mAP, progress,
                                     status="running", metrics_history=None, error_message=None,
                                     checkpoint_path=None):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE training_jobs SET
                    current_epoch = ?,
                    train_loss = ?,
                    val_loss = ?,
                    train_acc = ?,
                    val_acc = ?,
                    precision = ?,
                    recall = ?,
                    mAP = ?,
                    progress_percent = ?,
                    status = ?,
                    metrics_history = COALESCE(?, metrics_history),
                    error_message = COALESCE(?, error_message),
                    checkpoint_path = COALESCE(?, checkpoint_path),
                    completed_at = CASE WHEN ? IN ('completed', 'failed', 'stopped') THEN ? ELSE completed_at END
                WHERE id = ?
            """, (
                current_epoch, train_loss, val_loss, train_acc, val_acc,
                precision, recall, mAP, progress, status,
                json.dumps(metrics_history) if metrics_history is not None else None,
                error_message, checkpoint_path,
                status, now_str, job_id
            ))
            conn.commit()

    def get_training_job(self, job_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            try:
                d["metrics_history"] = json.loads(d.get("metrics_history") or "[]")
            except Exception:
                d["metrics_history"] = []
            return d

    def get_all_training_jobs(self, limit=20):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            res = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["metrics_history"] = json.loads(d.get("metrics_history") or "[]")
                except Exception:
                    d["metrics_history"] = []
                res.append(d)
            return res

    def add_training_log(self, job_id, epoch, message, log_level="INFO"):
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_logs (job_id, epoch, log_level, message, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (job_id, epoch, log_level, message, now_str))
            conn.commit()

    def get_training_logs(self, job_id, limit=100):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_logs WHERE job_id = ? ORDER BY id ASC LIMIT ?", (job_id, limit))
            return [dict(r) for r in cursor.fetchall()]

    def create_model_version(self, version_tag, job_id, task_type, architecture, weights_path, metrics):
        m_id = f"MOD_{version_tag}_{os.urandom(2).hex().upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_json = json.dumps(metrics) if isinstance(metrics, dict) else str(metrics)

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_versions (
                    id, version_tag, job_id, task_type, architecture, weights_path, metrics, is_active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (m_id, version_tag, job_id, task_type, architecture, weights_path, metrics_json, now_str))
            # Deactivate previous models for the same task
            cursor.execute("UPDATE model_versions SET is_active = 0 WHERE id != ? AND task_type = ?", (m_id, task_type))
            conn.commit()

        return m_id

    def get_model_versions(self, task_type=None):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if task_type:
                cursor.execute("SELECT * FROM model_versions WHERE task_type = ? ORDER BY created_at DESC", (task_type,))
            else:
                cursor.execute("SELECT * FROM model_versions ORDER BY created_at DESC")
            res = []
            for r in cursor.fetchall():
                d = dict(r)
                try:
                    d["metrics"] = json.loads(d.get("metrics") or "{}")
                except Exception:
                    d["metrics"] = {}
                res.append(d)
            return res

    def set_active_model_version(self, version_id):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_versions WHERE id = ?", (version_id,))
            target = cursor.fetchone()
            if not target:
                return False
            task = target["task_type"]
            cursor.execute("UPDATE model_versions SET is_active = 0 WHERE task_type = ?", (task,))
            cursor.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (version_id,))
            conn.commit()
            return True

    # =========================================================================
    # MODULE 5: CAMERA DEVICE INTEGRATION
    # =========================================================================

    def get_camera_devices(self):
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM camera_devices ORDER BY id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def update_camera_status(self, cam_id, status="ONLINE", fps=0.0, latency_ms=0):
        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE camera_devices SET status = ?, fps = ?, latency_ms = ? WHERE id = ?
            """, (status, fps, latency_ms, cam_id))
            conn.commit()

    # =========================================================================
    # MODULE 6: TEACHER-REVIEWED ATTENDANCE & AUDIT TRAIL
    # =========================================================================

    def get_teacher_attendance_roster(self, session_id=None, class_name=None, section=None):
        """Returns class roster merged with verified attendance status for active session."""
        sess = session_id
        if not sess:
            active_s = self.get_active_session()
            sess = active_s["session_id"] if active_s else None

        students = self.get_all_students(section=section)
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            att_map = {}
            if sess:
                cursor.execute("SELECT * FROM attendance_records WHERE session_id = ?", (sess,))
                for r in cursor.fetchall():
                    att_map[r["student_id"]] = dict(r)
                # Fallback check on base attendance table
                cursor.execute("SELECT * FROM attendance WHERE session_id = ?", (sess,))
                for r in cursor.fetchall():
                    if r["roll_number"] not in att_map:
                        att_map[r["roll_number"]] = dict(r)

        roster = []
        for s in students:
            sid = s.get("student_id") or s.get("roll_number")
            att = att_map.get(sid) or att_map.get(s.get("roll_number"))
            item = dict(s)
            item["attendance_status"] = att["status"] if att else "ABSENT"
            item["first_check_in"] = att["first_check_in"] if att else None
            item["verified_by"] = att.get("verified_by") if att else None
            item["audit_notes"] = att.get("audit_notes") if att else None
            roster.append(item)

        return roster

    def teacher_verify_attendance(self, session_id, student_id, status="PRESENT",
                                  verified_by="Teacher", audit_notes="Teacher Manual Review"):
        """Marks or updates attendance with audit trail and duplicate protection."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        att_id = f"ATT_{session_id}_{student_id}"

        student = self.get_student(student_id)
        st_name = student.get("name", "Student") if student else "Student"
        roll = student.get("roll_number", student_id) if student else student_id

        with sqlite3.connect(SQLITE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO attendance_records (
                    attendance_id, session_id, student_id, roll_number, student_name,
                    status, first_check_in, last_seen, total_duration_minutes,
                    verification_type, verified_by, audit_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 45, 'TEACHER_REVIEW', ?, ?, ?)
                ON CONFLICT(session_id, student_id) DO UPDATE SET
                    status = excluded.status,
                    verified_by = excluded.verified_by,
                    audit_notes = excluded.audit_notes,
                    last_seen = excluded.last_seen
            """, (
                att_id, session_id, student_id, roll, st_name,
                status.upper(), now_str, now_str, verified_by, audit_notes, now_str
            ))

            # Mirror into primary attendance table
            cursor.execute("""
                INSERT INTO attendance (
                    attendance_id, session_id, roll_number, name, status,
                    first_check_in, last_seen, total_duration_minutes, verification_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 45.0, 'TEACHER_REVIEW')
                ON CONFLICT(session_id, roll_number) DO UPDATE SET
                    status = excluded.status,
                    last_seen = excluded.last_seen
            """, (
                att_id, session_id, roll, st_name, status.upper(),
                now_str, now_str
            ))

            # Record audit event
            event_id = f"EVT_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(2).hex()}"
            cursor.execute("""
                INSERT INTO session_events (event_id, session_id, roll_number, name, event_type, details, timestamp)
                VALUES (?, ?, ?, ?, 'MANUAL_VERIFY', ?, ?)
            """, (event_id, session_id, roll, st_name, f"Status set to {status.upper()} by {verified_by}. Note: {audit_notes}", now_str))

            conn.commit()

        return True

    def generate_attendance_report(self, date_str=None, month_str=None):
        """Generates summary attendance reports for daily or monthly review."""
        with sqlite3.connect(SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if date_str:
                cursor.execute("""
                    SELECT ar.*, cs.subject, cs.section, cs.faculty_name
                    FROM attendance_records ar
                    LEFT JOIN classroom_sessions cs ON ar.session_id = cs.session_id
                    WHERE date(ar.created_at) = date(?)
                    ORDER BY ar.created_at DESC
                """, (date_str,))
            elif month_str:
                cursor.execute("""
                    SELECT ar.*, cs.subject, cs.section, cs.faculty_name
                    FROM attendance_records ar
                    LEFT JOIN classroom_sessions cs ON ar.session_id = cs.session_id
                    WHERE strftime('%Y-%m', ar.created_at) = ?
                    ORDER BY ar.created_at DESC
                """, (month_str,))
            else:
                cursor.execute("""
                    SELECT ar.*, cs.subject, cs.section, cs.faculty_name
                    FROM attendance_records ar
                    LEFT JOIN classroom_sessions cs ON ar.session_id = cs.session_id
                    ORDER BY ar.created_at DESC LIMIT 200
                """)
            return [dict(r) for r in cursor.fetchall()]


# Global singleton instance
db_manager = DatabaseManager()
