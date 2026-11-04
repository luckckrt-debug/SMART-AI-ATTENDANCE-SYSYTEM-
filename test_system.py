import unittest
import os
import sys
import json
import base64
import time
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database_manager import db_manager
from classroom_camera import classroom_camera
from app import app


class TestSmartClassroomAttendanceSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.test_roll = "8888"
        cls.test_name = "TestStudent"

    @classmethod
    def tearDownClass(cls):
        db_manager.delete_student(cls.test_roll)

    # -------------------------------------------------------------
    # 1. FILE & MODEL ASSET VERIFICATION
    # -------------------------------------------------------------
    def test_01_required_files_and_models(self):
        """Verify presence of FaceNet, Dlib landmarks, Haar cascade, and critical modules."""
        required = [
            BASE_DIR / "PreTrained_model" / "facenet_keras.h5",
            BASE_DIR / "shape_predictor_68_face_landmarks.dat",
            BASE_DIR / "FaceDetection" / "faces.xml",
            BASE_DIR / "classroom_camera.py",
            BASE_DIR / "database_manager.py",
            BASE_DIR / "app.py",
            BASE_DIR / "run.bat",
            BASE_DIR / "run.py",
        ]
        for p in required:
            self.assertTrue(p.exists(), f"Critical file missing: {p.name}")

    # -------------------------------------------------------------
    # 2. SESSION CREATION & SECTION CHANGE
    # -------------------------------------------------------------
    def test_02_session_creation_and_section_switch(self):
        """Verify classroom session creation, section switching, and active session management."""
        # Create initial session for Section A
        sess_a = db_manager.create_session(
            academic_year="2026-2027",
            department="Computer Science & Engineering",
            class_name="B.Tech 3rd Year",
            section="Section A",
            subject="Neural Networks & Deep Learning",
            faculty_name="Dr. Alan Turing"
        )
        self.assertTrue(sess_a.startswith("SESS_"))

        active = db_manager.get_active_session()
        self.assertIsNotNone(active)
        self.assertEqual(active["session_id"], sess_a)
        self.assertEqual(active["section"], "Section A")
        self.assertEqual(active["status"], "ACTIVE")

        # Switch section: create session for Section B
        sess_b = db_manager.create_session(
            academic_year="2026-2027",
            department="Information Technology",
            class_name="B.Tech 3rd Year",
            section="Section B",
            subject="Robotics & Vision",
            faculty_name="Prof. Ada Lovelace"
        )
        self.assertTrue(sess_b.startswith("SESS_"))

        # Verify old session A was closed and new session B is active
        sess_a_data = db_manager.get_session(sess_a)
        self.assertEqual(sess_a_data["status"], "COMPLETED")
        self.assertIsNotNone(sess_a_data["session_end_time"])

        active_now = db_manager.get_active_session()
        self.assertEqual(active_now["session_id"], sess_b)
        self.assertEqual(active_now["section"], "Section B")

    # -------------------------------------------------------------
    # 3. STUDENT PROFILE & CONSENT ENFORCEMENT
    # -------------------------------------------------------------
    def test_03_student_profile_management_and_consent(self):
        """Verify student enrollment with metadata, profile editing, and consent safeguards."""
        # Clean previous if exists
        db_manager.delete_student(self.test_roll)

        # Enroll student with informed consent
        success, msg = db_manager.enroll_student(
            roll_number=self.test_roll,
            name=self.test_name,
            department="Computer Science & Engineering",
            section="Section B",
            profile_photo="/static/profiles/8888.jpg",
            sync_csv=False
        )
        self.assertTrue(success, f"Enrollment failed: {msg}")

        student = db_manager.get_student(self.test_roll)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], self.test_name)
        self.assertEqual(student["consent_given"], 1)

        # Profile correction
        updated, u_msg = db_manager.update_student_profile(
            roll_number=self.test_roll,
            name="TestStudent Updated",
            department="AI & Robotics",
            section="Section B",
            enrollment_status="Active"
        )
        self.assertTrue(updated)
        student_updated = db_manager.get_student(self.test_roll)
        self.assertEqual(student_updated["name"], "TestStudent Updated")
        self.assertEqual(student_updated["department"], "AI & Robotics")

    # -------------------------------------------------------------
    # 4. DUPLICATE ATTENDANCE PREVENTION
    # -------------------------------------------------------------
    def test_04_session_attendance_and_duplicate_prevention(self):
        """Verify single attendance record per student per session, preventing duplicates on re-detection."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # First Check-In
        ok1, msg1, action1 = db_manager.record_session_presence(sess_id, self.test_roll)
        self.assertTrue(ok1, f"Initial presence recording failed: {msg1}")
        self.assertEqual(action1, "CHECK_IN")

        # Verify exactly 1 attendance record
        records_after_1 = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == self.test_roll]
        self.assertEqual(len(records_after_1), 1)
        first_check_in_time = records_after_1[0]["first_check_in"]

        # Simulate subsequent camera detection frame in same session
        time.sleep(1)
        ok2, msg2, action2 = db_manager.record_session_presence(sess_id, self.test_roll)
        self.assertTrue(ok2)
        self.assertEqual(action2, "ACTIVE")

        # Verify STILL exactly 1 record (no duplicate created)
        records_after_2 = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == self.test_roll]
        self.assertEqual(len(records_after_2), 1, "Duplicate attendance record was improperly created!")
        self.assertEqual(records_after_2[0]["first_check_in"], first_check_in_time)

    # -------------------------------------------------------------
    # 5. ENTRY, EXIT & RETURN EVENT MANAGEMENT
    # -------------------------------------------------------------
    def test_05_checkin_exit_and_return_lifecycle(self):
        """Verify student exit tracking, return event logging, and status transitions."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # Student leaves the classroom
        exit_ok, exit_msg = db_manager.record_student_exit(sess_id, self.test_roll, reason="Lab equipment fetch")
        self.assertTrue(exit_ok, f"Exit recording failed: {exit_msg}")

        # Verify status is CHECKED_OUT and exit_count == 1
        att_row = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == self.test_roll][0]
        self.assertEqual(att_row["status"], "CHECKED_OUT")
        self.assertEqual(att_row["exit_count"], 1)

        # Verify EXIT event in audit log
        events = db_manager.get_session_events(sess_id, limit=10)
        exit_events = [e for e in events if e["roll_number"] == self.test_roll and e["event_type"] == "EXIT"]
        self.assertGreaterEqual(len(exit_events), 1)

        # Student returns to the classroom
        ret_ok, ret_msg, ret_action = db_manager.record_session_presence(sess_id, self.test_roll)
        self.assertTrue(ret_ok)
        self.assertEqual(ret_action, "RETURN")

        # Verify status is back to PRESENT and still no duplicate record
        records = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == self.test_roll]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "PRESENT")

        # Verify RETURN event logged
        events_after = db_manager.get_session_events(sess_id, limit=10)
        return_events = [e for e in events_after if e["roll_number"] == self.test_roll and e["event_type"] == "RETURN"]
        self.assertGreaterEqual(len(return_events), 1)

    # -------------------------------------------------------------
    # 6. NEUTRAL NOTIFICATIONS ON REPEATED EXITS
    # -------------------------------------------------------------
    def test_06_neutral_notification_on_repeated_exits(self):
        """Verify neutral informational notice to teacher when repeated exits occur."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # Record second exit event for the same student
        exit2_ok, _ = db_manager.record_student_exit(sess_id, self.test_roll, reason="Library book reference")
        self.assertTrue(exit2_ok)

        # Check notifications
        notifs = db_manager.get_session_notifications(sess_id)
        student_notifs = [n for n in notifs if n["roll_number"] == self.test_roll]
        self.assertGreaterEqual(len(student_notifs), 1, "Expected neutral notification on repeated exits")
        self.assertEqual(student_notifs[0]["severity"], "NEUTRAL_NOTICE")
        self.assertIn("recorded 2 exit events", student_notifs[0]["message"])

        # Test marking notification read
        nid = student_notifs[0]["notification_id"]
        db_manager.mark_notification_read(nid)
        unread = db_manager.get_session_notifications(sess_id, unread_only=True)
        self.assertTrue(all(n["notification_id"] != nid for n in unread))

    # -------------------------------------------------------------
    # 7. SESSION DURATION ANALYTICS
    # -------------------------------------------------------------
    def test_07_session_duration_analytics(self):
        """Verify transparent presence duration computation."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # Student checks back in and presence is confirmed
        db_manager.record_session_presence(sess_id, self.test_roll)
        roster = db_manager.get_session_roster_status(sess_id)
        student_row = [r for r in roster if r["roll_number"] == self.test_roll][0]

        self.assertIn("duration_formatted", student_row)
        self.assertIn("m", student_row["duration_formatted"])
        self.assertIn("s", student_row["duration_formatted"])

    # -------------------------------------------------------------
    # 8. MULTI-PERSON CAMERA PROCESSING
    # -------------------------------------------------------------
    def test_08_multiple_person_camera_processing(self):
        """Verify multi-person frame processing, health metrics, and consent masking."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # Create synthetic multi-person frame (e.g., 480x640 with drawn face regions)
        synthetic_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Draw two mock face circles to simulate multiple subjects in frame
        cv2.circle(synthetic_frame, (160, 200), 60, (150, 150, 150), -1)
        cv2.circle(synthetic_frame, (480, 200), 60, (150, 150, 150), -1)

        annotated, detections = classroom_camera.process_frame(synthetic_frame, session_id=sess_id)
        self.assertIsNotNone(annotated)
        self.assertIsInstance(detections, list)

        health = classroom_camera.get_health()
        self.assertIn("fps", health)
        self.assertIn("status", health)
        self.assertIn("latency_ms", health)
        self.assertIn(health["status"], ["ACTIVE", "ONLINE", "STANDBY", "OFFLINE"])

    # -------------------------------------------------------------
    # 9. MANUAL TEACHER CORRECTION
    # -------------------------------------------------------------
    def test_09_manual_teacher_correction(self):
        """Verify instructor manual override capability for attendance status."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        corr_ok, msg = db_manager.manual_teacher_correction(
            session_id=sess_id,
            roll_number=self.test_roll,
            new_status="PRESENT",
            reason="Instructor verified in-person participation"
        )
        self.assertTrue(corr_ok, msg)

        # Verify status and audit trail
        records = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == self.test_roll]
        self.assertEqual(records[0]["status"], "PRESENT")
        self.assertEqual(records[0]["verification_method"], "MANUAL_CORRECTION")

        events = db_manager.get_session_events(sess_id, limit=5)
        override_events = [e for e in events if e["event_type"] == "MANUAL_OVERRIDE" and e["roll_number"] == self.test_roll]
        self.assertGreaterEqual(len(override_events), 1)

    # -------------------------------------------------------------
    # 10. CSV EXPORT & HTTP ENDPOINTS
    # -------------------------------------------------------------
    def test_10_csv_exports_and_flask_routes(self):
        """Verify session CSV export file generation and all application web endpoints."""
        active = db_manager.get_active_session()
        sess_id = active["session_id"]

        # Session CSV Export
        csv_path = db_manager.export_session_csv(sess_id)
        self.assertTrue(os.path.exists(csv_path), "Session CSV file was not exported")
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            self.assertIn("Roll Number", header)
            self.assertIn("Student Name", header)
            self.assertIn("Attendance Status", header)
            self.assertIn("Total Verified Presence", header)

        # Flask Endpoints Verification (Direct 200s and Redirect 302s)
        routes = [
            ("/", 200),
            ("/classroom", 200),
            ("/sessions", 302),
            ("/students", 302),
            ("/records", 200),
            (f"/records?session_id={sess_id}", 200),
            ("/enroll", 302),
            ("/train", 302),
            ("/attendance/Hindi", 302),
            ("/attendance/English", 302),
            ("/database-status", 302),
            ("/settings", 302),
            ("/api/status", 200),
            ("/api/train/status", 200),
            (f"/api/sessions/{sess_id}/notifications", 200),
            (f"/export/session_csv/{sess_id}", 200),
            ("/export/csv/Hindi", 200),
            ("/export/csv/English", 200),
            ("/export/csv/Enrollment", 200),
        ]

        for path, expected_code in routes:
            resp = self.client.get(path)
            self.assertEqual(
                resp.status_code,
                expected_code,
                f"Route {path} failed with HTTP {resp.status_code}"
            )

    # -------------------------------------------------------------
    # 11. RBAC & ADMIN PANEL SEPARATION VERIFICATION
    # -------------------------------------------------------------
    def test_11_rbac_and_admin_panel_separation(self):
        """Verify strict RBAC separation between Teacher Home and Admin Panel."""
        # 1. Accessing Admin Panel as Teacher should be strictly FORBIDDEN (HTTP 403)
        self.client.get("/api/auth/switch_role/teacher")
        resp_teacher = self.client.get("/admin")
        self.assertEqual(resp_teacher.status_code, 403, "Teacher was able to access /admin")
        self.assertIn(b"Access Restricted", resp_teacher.data)

        # 2. Accessing Admin Panel as Admin should succeed (HTTP 200)
        self.client.get("/api/auth/switch_role/admin")
        resp_admin = self.client.get("/admin")
        self.assertEqual(resp_admin.status_code, 200, "Admin could not access /admin")
        self.assertIn(b"Admin Panel", resp_admin.data)
        self.assertIn(b"System Administration & Configuration Console", resp_admin.data)

        # 3. Verify Teacher Dashboard only has Attendance and Session Manager
        self.client.get("/api/auth/switch_role/teacher")
        home_resp = self.client.get("/")
        self.assertEqual(home_resp.status_code, 200)
        self.assertIn(b"Classroom Attendance", home_resp.data)
        self.assertIn(b"Session Manager", home_resp.data)

    # -------------------------------------------------------------
    # 12. DATABASE PERSISTENCE & MONGODB ARCHITECTURE
    # -------------------------------------------------------------
    def test_12_database_persistence_and_mongodb_mode(self):
        """Verify database persistence architecture (MongoDB when reachable, SQLite fallback with full schema parity)."""
        status = db_manager.get_status()
        self.assertIn(status["mode"], ["MongoDB", "SQLite (Fallback)", "SQLite"])
        if status["is_mongo_connected"]:
            self.assertEqual(status["mode"], "MongoDB")
            self.assertIn("Connected to MongoDB", status["status_message"])
            self.assertIsNone(status["mongo_error"])
        else:
            self.assertTrue(os.path.exists(status["sqlite_path"]))
            self.assertIn("SQLite", status["status_message"])


if __name__ == "__main__":
    unittest.main()
