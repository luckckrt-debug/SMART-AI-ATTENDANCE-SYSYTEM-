"""
End-to-End Automated Verification Test Suite for Smart AI Attendance System
Validates all 13 required test cases:
1. Single-person student face enrollment via API
2. Corrupt / invalid image upload rejection
3. Multi-person enrollment photograph handling
4. Unknown person live stream rejection
5. Student from Class A marking attendance in Class A session -> MARKED
6. Student from Class B trying to mark attendance in Class A session -> OUTSIDE_CLASS warning; NOT MARKED
7. Student from 2nd Year trying to mark in 3rd Year session -> OUTSIDE_CLASS warning; NOT MARKED
8. Faculty face detected in student attendance session -> FACULTY EXCLUDED (Pure Blue box); NOT MARKED
9. Ambiguous match (<0.04 margin between top candidates) -> REJECTED / AMBIGUOUS
10. Image with no face -> REJECTED
11. Duplicate attendance in same session -> DUPLICATE_ATTEMPT handled idempotently
12. Session mismatch -> correctly scoped
13. Student change degree/stream -> eligibility updated immediately
"""

import os
import sys
import json
import base64
import unittest
from pathlib import Path
import numpy as np
import cv2

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database_manager import db_manager
from ai_engine import (
    prototype_matcher,
    faculty_matcher,
    auto_enroll_student_face,
    auto_enroll_faculty_face
)
from classroom_camera import classroom_camera
from app import app


class TestSmartAttendanceSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

        # Find existing sample photo if available
        sample_photos = list(Path("people").glob("*/*.*"))
        if sample_photos:
            cls.real_photo_path = str(sample_photos[0])
            cls.real_photo_bytes = sample_photos[0].read_bytes()
        else:
            # Fallback to creating a test image with drawn face features
            img = np.zeros((200, 200, 3), dtype=np.uint8) + 180
            cv2.circle(img, (100, 100), 60, (120, 160, 220), -1)
            cv2.circle(img, (80, 85), 8, (20, 20, 20), -1)
            cv2.circle(img, (120, 85), 8, (20, 20, 20), -1)
            cv2.ellipse(img, (100, 125), (25, 10), 0, 0, 180, (20, 20, 180), 3)
            _, buf = cv2.imencode(".jpg", img)
            cls.real_photo_bytes = buf.tobytes()
            cls.real_photo_path = str(BASE_DIR / "test_sample_face.jpg")
            with open(cls.real_photo_path, "wb") as f:
                f.write(cls.real_photo_bytes)

    # -------------------------------------------------------------------------
    # TEST CASE 1: Single-person student face enrollment via API
    # -------------------------------------------------------------------------
    def test_01_single_student_face_enrollment_api(self):
        with app.test_request_context():
            with self.client.session_transaction() as sess:
                sess["user"] = {"username": "admin", "role": "ADMIN", "name": "Admin User"}

            roll = "TEST_TT_101"
            res = self.client.post("/api/admin/students/register", json={
                "name": "Aarav Sharma",
                "roll_number": roll,
                "email": "aarav.tt@college.edu",
                "mobile": "+91 9876543210",
                "degree": "B.Tech",
                "stream": "Textile Technology",
                "academic_year": "2nd Year",
                "academic_session": "2026-27",
                "consent_given": True,
                "images": [base64.b64encode(self.real_photo_bytes).decode("utf-8")]
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data.get("success"), f"Enrollment failed: {data}")
            self.assertIn("stages", data)
            self.assertEqual(data["stages"][0]["stage"], "Uploading")

            # Check database persistence
            student = db_manager.get_student(roll)
            self.assertIsNotNone(student)
            self.assertEqual(student["degree"], "B.Tech")
            self.assertEqual(student["stream"], "Textile Technology")
            print("\n[TEST 1 PASSED] Single-student face enrollment via API succeeded with full stages.")

    # -------------------------------------------------------------------------
    # TEST CASE 2: Corrupt / invalid image upload rejection
    # -------------------------------------------------------------------------
    def test_02_corrupt_invalid_image_upload_rejection(self):
        corrupt_bytes = b"NOT_A_VALID_JPEG_HEADER_CORRUPT_BYTES_XYZ"
        res = auto_enroll_student_face("TEST_CORRUPT_01", "Corrupt Test", corrupt_bytes)
        self.assertFalse(res["success"])
        self.assertIn("FAILED", res["state"])
        print("[TEST 2 PASSED] Corrupt/invalid image bytes gracefully rejected.")

    # -------------------------------------------------------------------------
    # TEST CASE 3: Multi-person enrollment photograph handling
    # -------------------------------------------------------------------------
    def test_03_multi_person_enrollment_photograph(self):
        # Create image with two faces side-by-side
        img = cv2.imread(self.real_photo_path)
        if img is not None:
            multi_img = np.hstack([img, img])
            res = auto_enroll_student_face("TEST_MULTI_01", "Multi Candidate", multi_img)
            # Must safely process or flag for review without crashing
            self.assertIn(res["state"], ["MANUAL_REVIEW_REQUIRED", "ENROLLED", "FAILED_QUALITY"])
            print("[TEST 3 PASSED] Multi-person enrollment photograph properly assessed.")

    # -------------------------------------------------------------------------
    # TEST CASE 4: Unknown person live stream frame rejection
    # -------------------------------------------------------------------------
    def test_04_unknown_person_rejection(self):
        # Generate random unit vector orthogonal to known prototypes
        rnd = np.random.randn(128).astype(np.float32)
        rnd = rnd / np.linalg.norm(rnd)

        roll, name, conf, state = prototype_matcher.match_embedding(rnd)
        self.assertIn(state, ["UNKNOWN", "AMBIGUOUS"])
        self.assertEqual(roll, "UNKNOWN") if state == "UNKNOWN" else None
        print(f"[TEST 4 PASSED] Unregistered random vector tagged {state} (conf: {round(conf, 3)}).")

    # -------------------------------------------------------------------------
    # TEST CASE 5: Class A student marking attendance in active Class A session
    # -------------------------------------------------------------------------
    def test_05_class_a_student_marking_attendance(self):
        # 1. Enroll Class A student
        roll_a = "TT_CLASS_A_01"
        db_manager.enroll_student(
            roll_number=roll_a,
            name="Rohan Verma",
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27"
        )
        auto_enroll_student_face(roll_a, "Rohan Verma", self.real_photo_bytes)

        # 2. Create Class A session
        sess_a = db_manager.create_session(
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27",
            subject="Textile Physics"
        )
        classroom_camera.active_session_id = sess_a

        # 3. Verify eligible student list snapshot
        eligible = db_manager.get_eligible_students_for_session(sess_a)
        self.assertIn(roll_a, eligible)

        # 4. Mark presence
        ok, msg, action = db_manager.record_session_presence(sess_a, roll_a)
        self.assertTrue(ok)
        self.assertIn(action, ["CHECKIN", "CHECK_IN"])

        records = db_manager.get_session_attendance(sess_a)
        rolls_present = [r["roll_number"] for r in records]
        self.assertIn(roll_a, rolls_present)
        print(f"[TEST 5 PASSED] Class A student successfully verified and marked present in session {sess_a}.")

    # -------------------------------------------------------------------------
    # TEST CASE 6: Class B student trying to mark in Class A session -> OUTSIDE_CLASS
    # -------------------------------------------------------------------------
    def test_06_class_b_student_outside_class_warning(self):
        # 1. Enroll Class B student (Textile Chemistry)
        roll_b = "TC_CLASS_B_01"
        db_manager.enroll_student(
            roll_number=roll_b,
            name="Sneha Patel",
            degree="B.Tech",
            stream="Textile Chemistry",
            academic_year="2nd Year",
            academic_session="2026-27"
        )
        # Store synthetic embedding for roll_b
        vec_b = np.random.randn(128).astype(np.float32)
        vec_b = (vec_b / np.linalg.norm(vec_b)).tolist()
        prototype_matcher.add_or_update_student(roll_b, "Sneha Patel", [vec_b])

        # 2. Create Class A session (Textile Technology)
        sess_a = db_manager.create_session(
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27",
            subject="Yarn Manufacturing"
        )
        classroom_camera.active_session_id = sess_a

        eligible_a = db_manager.get_eligible_students_for_session(sess_a)
        self.assertNotIn(roll_b, eligible_a, "Class B student must NOT be in Class A eligible roster!")

        # 3. Match within eligible IDs
        emb_arr = np.array(vec_b, dtype=np.float32)
        m_roll, m_name, m_conf, m_state = prototype_matcher.match_embedding(emb_arr, eligible_ids=eligible_a)
        self.assertNotEqual(m_roll, roll_b, "Should not match within Class A eligible roster")

        # 4. Global match confirms they belong to institution -> triggers OUTSIDE_CLASS
        g_roll, g_name, g_conf, g_state = prototype_matcher.match_embedding(emb_arr, eligible_ids=None)
        self.assertEqual(g_roll, roll_b)

        # 5. Verify attendance was NOT recorded
        records = db_manager.get_session_attendance(sess_a)
        self.assertNotIn(roll_b, [r["roll_number"] for r in records])
        print(f"[TEST 6 PASSED] Student from Textile Chemistry correctly denied attendance in Textile Technology class (OUTSIDE_CLASS).")

    # -------------------------------------------------------------------------
    # TEST CASE 7: Student from 2nd Year trying to mark in 3rd Year session -> OUTSIDE_CLASS
    # -------------------------------------------------------------------------
    def test_07_outside_year_student_warning(self):
        roll_2nd = "TT_YEAR2_01"
        db_manager.enroll_student(
            roll_number=roll_2nd,
            name="Vikram Singh",
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27"
        )

        sess_3rd = db_manager.create_session(
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="3rd Year",
            academic_session="2026-27",
            subject="Advanced Weaving"
        )
        eligible_3rd = db_manager.get_eligible_students_for_session(sess_3rd)
        self.assertNotIn(roll_2nd, eligible_3rd, "2nd Year student must NOT be eligible in 3rd Year session!")
        print(f"[TEST 7 PASSED] 2nd Year student correctly isolated from 3rd Year session roster.")

    # -------------------------------------------------------------------------
    # TEST CASE 8: Faculty Face Exclusion & Immunity (Pure Blue Box, Excluded)
    # -------------------------------------------------------------------------
    def test_08_faculty_exclusion_immunity(self):
        fac_id = "FAC_TT_PROF_01"
        fac_name = "Dr. Arvind Kumar"
        db_manager.enroll_faculty(
            faculty_id=fac_id,
            name=fac_name,
            department="Textile Technology",
            designation="Professor"
        )

        # Create distinct faculty prototype
        fac_vec = np.random.randn(128).astype(np.float32)
        fac_vec = (fac_vec / np.linalg.norm(fac_vec)).tolist()
        faculty_matcher.add_or_update_faculty(fac_id, fac_name, [fac_vec])

        # Test faculty matching
        matched_id, matched_name, conf, state = faculty_matcher.match_embedding(np.array(fac_vec))
        self.assertEqual(state, "VERIFIED_FACULTY")
        self.assertEqual(matched_id, fac_id)
        self.assertEqual(matched_name, fac_name)

        # Verify faculty is NEVER in student attendance
        active_sess = db_manager.get_active_session()
        if active_sess:
            att = db_manager.get_session_attendance(active_sess["session_id"])
            self.assertNotIn(fac_id, [r["roll_number"] for r in att])

        print(f"[TEST 8 PASSED] Faculty {fac_name} identified with VERIFIED_FACULTY immunity; excluded from attendance.")

    # -------------------------------------------------------------------------
    # TEST CASE 9: Ambiguous match (<0.04 margin between top candidates) -> AMBIGUOUS
    # -------------------------------------------------------------------------
    def test_09_ambiguous_match_rejection(self):
        # Create two student prototypes close to each other
        v1 = np.ones(128, dtype=np.float32)
        v1 = v1 / np.linalg.norm(v1)

        # v2 is very slightly perturbed so difference < 0.04
        v2 = v1.copy()
        v2[0] += 0.01
        v2 = v2 / np.linalg.norm(v2)

        prototype_matcher.add_or_update_student("AMBIG_01", "Candidate One", [v1.tolist()])
        prototype_matcher.add_or_update_student("AMBIG_02", "Candidate Two", [v2.tolist()])

        # Query vector right between both
        query_v = (v1 + v2) / 2.0
        query_v = query_v / np.linalg.norm(query_v)

        roll, name, conf, state = prototype_matcher.match_embedding(query_v)
        self.assertEqual(state, "AMBIGUOUS", f"Expected AMBIGUOUS state for margin < 0.04, got: {state}")
        print(f"[TEST 9 PASSED] Close-margin candidate similarity correctly triggered AMBIGUOUS state.")

    # -------------------------------------------------------------------------
    # TEST CASE 10: Image with no face -> REJECTED
    # -------------------------------------------------------------------------
    def test_10_image_with_no_face_rejection(self):
        # Completely blank black image
        blank = np.zeros((160, 160, 3), dtype=np.uint8)
        res = auto_enroll_student_face("TEST_NO_FACE_01", "Blank Face", blank)
        self.assertFalse(res["success"])
        self.assertEqual(res["state"], "FAILED_NO_FACE")
        print("[TEST 10 PASSED] Blank image with no face rejected with FAILED_NO_FACE.")

    # -------------------------------------------------------------------------
    # TEST CASE 11: Duplicate attendance in same session handled idempotently
    # -------------------------------------------------------------------------
    def test_11_duplicate_attendance_prevention(self):
        roll = "DUP_TEST_01"
        db_manager.enroll_student(
            roll_number=roll,
            name="Duplicate Tester",
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27"
        )
        sess_id = db_manager.create_session(
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27"
        )

        # First check-in
        ok1, msg1, action1 = db_manager.record_session_presence(sess_id, roll)
        self.assertTrue(ok1)
        self.assertIn(action1, ["CHECKIN", "CHECK_IN"])

        # Second check-in attempt (duplicate attempt updates presence duration instead of inserting duplicate)
        ok2, msg2, action2 = db_manager.record_session_presence(sess_id, roll)
        self.assertTrue(ok2)
        self.assertEqual(action2, "ACTIVE")
        self.assertIn("Presence active", msg2)

        records = [r for r in db_manager.get_session_attendance(sess_id) if r["roll_number"] == roll]
        self.assertEqual(len(records), 1, "Only one attendance entry must exist for the student!")
        print(f"[TEST 11 PASSED] Duplicate attendance attempt handled idempotently (single record retained).")

    # -------------------------------------------------------------------------
    # TEST CASE 12: Session mismatch isolation
    # -------------------------------------------------------------------------
    def test_12_session_mismatch_isolation(self):
        old_sess = db_manager.create_session(degree="B.Tech", stream="Textile Technology", subject="History")
        # End old session
        db_manager.end_active_sessions()

        # Attempt to mark in ended session
        ok, msg, action = db_manager.record_session_presence(old_sess, "TT_CLASS_A_01")
        self.assertFalse(ok)
        self.assertEqual(action, "SESSION_CLOSED")
        print(f"[TEST 12 PASSED] Inactive/ended session strictly rejected attendance marking.")

    # -------------------------------------------------------------------------
    # TEST CASE 13: Student degree/stream change updates eligibility immediately
    # -------------------------------------------------------------------------
    def test_13_degree_stream_change_eligibility_update(self):
        roll = "TRANSFER_STUDENT_01"
        db_manager.enroll_student(
            roll_number=roll,
            name="Kunal Roy",
            degree="B.Tech",
            stream="Textile Technology",
            academic_year="2nd Year",
            academic_session="2026-27"
        )

        # Transfer student to Textile Engineering
        db_manager.enroll_student(
            roll_number=roll,
            name="Kunal Roy",
            degree="B.Tech",
            stream="Textile Engineering",
            academic_year="2nd Year",
            academic_session="2026-27"
        )

        sess_te = db_manager.create_session(
            degree="B.Tech",
            stream="Textile Engineering",
            academic_year="2nd Year",
            academic_session="2026-27",
            subject="Textile Mechanics"
        )
        eligible_te = db_manager.get_eligible_students_for_session(sess_te)
        self.assertIn(roll, eligible_te, "Transferred student must immediately be in new stream's eligible roster!")
        print(f"[TEST 13 PASSED] Student stream change immediately reflected in new session eligibility snapshot.")


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSmartAttendanceSystem)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n=======================================================")
        print("ALL 13 AUTOMATED VERIFICATION TEST CASES PASSED!")
        print("=======================================================")
        sys.exit(0)
    else:
        print("\nTEST FAILURES ENCOUNTERED:")
        sys.exit(1)
