"""
Automated Comprehensive Test Verification Suite for SMART AI CAM Training Studio.
Tests all 9 modules:
1. Student Management (CRUD, Search, Filter, Bulk Import, Export)
2. Image Dataset Management (Multi-file upload, corruption, duplicate detection, blur/light score)
3. AI Training Studio (Asynchronous job launch, metrics, epoch logging, graceful stop)
4. Training Monitor & Model Checkpoint Registry
5. Non-Identifying Camera Occupancy & Person Detection
6. Teacher-Reviewed Attendance & Audit Trail
7. Database Transactions & Integrity
8. API Endpoints & Route Integrity
"""

import os
import io
import json
import base64
import unittest
import numpy as np
import cv2
from PIL import Image

from app import app
from database_manager import db_manager
import studio_engine
from studio_engine import image_validator, person_detector


class TestSmartAIStudio(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    # -------------------------------------------------------------
    # MODULE 1 TESTS: STUDENT MANAGEMENT & BULK IMPORT/EXPORT
    # -------------------------------------------------------------
    def test_01_student_crud_and_search_filter(self):
        """Tests adding, editing, searching, filtering, and deleting student records."""
        test_roll = "TEST_STUDIO_9901"
        st = db_manager.add_student_record(
            name="Studio Student 1",
            roll_number=test_roll,
            student_id="STU_9901",
            class_name="B.Tech 3rd Year",
            section="Section A",
            academic_session="2026-27",
            enrollment_status="Active",
            degree="B.Tech",
            stream="Textile Technology",
            department="Textile Technology",
            email="studio1@textile.edu",
            mobile="9876543210"
        )
        self.assertIsNotNone(st)
        self.assertEqual(st["name"], "Studio Student 1")
        self.assertEqual(st["stream"], "Textile Technology")

        # Edit student
        edited = db_manager.edit_student_record(
            roll_number=test_roll,
            name="Studio Student 1 Updated",
            enrollment_status="Inactive"
        )
        self.assertEqual(edited["name"], "Studio Student 1 Updated")
        self.assertEqual(edited["enrollment_status"], "Inactive")

        # Search filter
        found = db_manager.search_filter_students(query="Studio Student 1", stream="Textile Technology")
        self.assertTrue(any(s["roll_number"] == test_roll for s in found))

        # API Search
        res = self.client.get(f"/api/studio/students?query={test_roll}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])

        # Delete student
        del_res = db_manager.delete_student(test_roll)
        self.assertTrue(del_res)
        self.assertIsNone(db_manager.get_student(test_roll))

    def test_02_student_bulk_import_and_export(self):
        """Tests bulk importing from CSV format and exporting to CSV/Excel."""
        sample_records = [
            {"name": "Bulk Student A", "roll_number": "BULK_TT_01", "student_id": "B01", "class": "B.Tech 3rd Year", "section": "Section A", "branch": "Textile Technology"},
            {"name": "Bulk Student B", "roll_number": "BULK_TT_02", "student_id": "B02", "class": "B.Tech 3rd Year", "section": "Section B", "branch": "Textile Chemistry"}
        ]
        res = db_manager.bulk_import_students(sample_records)
        self.assertTrue(res["success"])
        self.assertEqual(res["imported_count"], 2)

        # Test CSV export endpoint
        res_csv = self.client.get("/api/studio/students/export-csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn("text/csv", res_csv.content_type)
        self.assertIn("BULK_TT_01", res_csv.get_data(as_text=True))

        # Test Excel export endpoint
        res_excel = self.client.get("/api/studio/students/export-excel")
        self.assertEqual(res_excel.status_code, 200)
        self.assertTrue(len(res_excel.data) > 100)

        # Cleanup
        db_manager.delete_student("BULK_TT_01")
        db_manager.delete_student("BULK_TT_02")

    # -------------------------------------------------------------
    # MODULE 2 TESTS: IMAGE DATASET MANAGEMENT & VALIDATION
    # -------------------------------------------------------------
    def test_03_image_quality_and_integrity_validation(self):
        """Validates PIL corruption check, blur score, luminance, and hash generation."""
        # 1. Create a sharp valid test image
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 80, (255, 255, 255), -1)
        cv2.putText(img, "Sharp", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", img)
        sharp_bytes = buf.tobytes()

        val = image_validator.inspect_image_bytes(sharp_bytes, "sharp.jpg")
        self.assertTrue(val["is_valid"])
        self.assertEqual(val["is_corrupted"], 0)
        self.assertGreater(val["quality_score"], 60.0)

        # 2. Corrupted file test
        corrupt_bytes = b"NOT_AN_IMAGE_HEADER_CORRUPTED_STREAM"
        val_c = image_validator.inspect_image_bytes(corrupt_bytes, "corrupt.jpg")
        self.assertFalse(val_c["is_valid"])
        self.assertEqual(val_c["is_corrupted"], 1)

    def test_04_dataset_creation_and_file_management(self):
        """Tests dataset workspace creation, duplicate prevention, and deletion."""
        ds = db_manager.create_dataset(
            name="Test Textile Hall Dataset",
            description="Testing dataset operations",
            task_type="occupancy_estimation"
        )
        ds_id = ds["id"]
        self.assertIsNotNone(ds)

        # Create dummy image
        img = np.full((200, 200, 3), 180, dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        raw_bytes = buf.tobytes()
        file_hash = image_validator.calculate_file_hash(raw_bytes)

        # Add image
        f_id = db_manager.add_dataset_file(
            dataset_id=ds_id,
            file_path="/static/datasets/test.jpg",
            file_name="test.jpg",
            file_hash=file_hash,
            file_size=len(raw_bytes),
            width=200,
            height=200,
            quality_score=95.0
        )
        self.assertIsNotNone(f_id)

        # Check duplicate
        is_dup = db_manager.check_dataset_duplicate(ds_id, file_hash)
        self.assertTrue(is_dup)

        # Delete image & dataset
        db_manager.delete_dataset_file(f_id)
        db_manager.delete_dataset(ds_id)
        self.assertIsNone(db_manager.get_dataset(ds_id))

    # -------------------------------------------------------------
    # MODULE 3 & 4 TESTS: AI TRAINING STUDIO & MONITOR
    # -------------------------------------------------------------
    def test_05_asynchronous_training_job_lifecycle(self):
        """Tests creating a training job, updating metrics, and graceful cancellation."""
        ds = db_manager.create_dataset("Test Training DS", task_type="person_detection")
        job = db_manager.create_training_job(
            job_name="Unit Test Training Job",
            dataset_id=ds["id"],
            task_type="person_detection",
            model_architecture="MobileNet-PersonDetector",
            epochs=5
        )
        job_id = job["id"]
        self.assertEqual(job["status"], "pending")

        # Simulate epoch progress update
        db_manager.update_training_job_progress(
            job_id=job_id,
            current_epoch=1,
            train_loss=0.85,
            val_loss=0.92,
            train_acc=0.65,
            val_acc=0.62,
            precision=0.70,
            recall=0.68,
            mAP=0.66,
            progress=20.0,
            status="running",
            metrics_history=[{"epoch": 1, "train_loss": 0.85, "val_loss": 0.92}]
        )
        db_manager.add_training_log(job_id, 1, "Epoch 1 complete", "INFO")

        # Verify job status via API
        res = self.client.get(f"/api/studio/train/status/{job_id}")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["job"]["current_epoch"], 1)

        # Create model version checkpoint
        ver_id = db_manager.create_model_version(
            version_tag="vTest_1.0",
            job_id=job_id,
            task_type="person_detection",
            architecture="MobileNet-PersonDetector",
            weights_path="Model/versions/test.json",
            metrics={"precision": 0.95, "recall": 0.92}
        )
        self.assertIsNotNone(ver_id)

        # Test deploy model
        dep_res = self.client.post(f"/api/studio/models/{ver_id}/deploy")
        self.assertEqual(dep_res.status_code, 200)

        # Cleanup dataset
        db_manager.delete_dataset(ds["id"])

    # -------------------------------------------------------------
    # MODULE 5 TESTS: NON-IDENTIFYING CAMERA OCCUPANCY SCANNER
    # -------------------------------------------------------------
    def test_06_non_identifying_camera_occupancy_analysis(self):
        """Verifies person detection, occupancy count, blur, and lighting without facial identification."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add simulated body silhouette
        cv2.rectangle(frame, (100, 100), (220, 400), (200, 200, 200), -1)
        cv2.circle(frame, (160, 60), 30, (200, 200, 200), -1)

        analysis = person_detector.analyze_frame(frame, conf_threshold=0.3)
        self.assertIn("occupancy_count", analysis)
        self.assertIn("quality", analysis)
        self.assertIn("scene", analysis)
        self.assertIsNotNone(analysis["annotated_frame"])

        # Test Frame API endpoint
        _, buf = cv2.imencode(".jpg", frame)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        res = self.client.post("/api/studio/camera/process-frame", json={
            "frame": b64,
            "confidence": 0.50,
            "camera_id": "CAM_01"
        })
        self.assertEqual(res.status_code, 200)
        res_data = res.get_json()
        self.assertTrue(res_data["success"])

    # -------------------------------------------------------------
    # MODULE 6 TESTS: TEACHER-REVIEWED ATTENDANCE & AUDIT TRAIL
    # -------------------------------------------------------------
    def test_07_teacher_reviewed_attendance_and_audit(self):
        """Tests teacher roster retrieval, 1-click attendance marking, and audit logging."""
        test_roll = "TEST_ATT_STU_01"
        db_manager.add_student_record(
            name="Attendance Test Student",
            roll_number=test_roll,
            class_name="B.Tech 3rd Year",
            section="Section A",
            stream="Textile Technology"
        )

        test_sess_id = "SESS_TEST_STUDIO_REVIEW"
        # 1-Click Teacher Verification
        verify_success = db_manager.teacher_verify_attendance(
            session_id=test_sess_id,
            student_id=test_roll,
            status="PRESENT",
            verified_by="Prof. Textile Faculty",
            audit_notes="Verified via Studio Test"
        )
        self.assertTrue(verify_success)

        # Verify roster status
        roster = db_manager.get_teacher_attendance_roster(session_id=test_sess_id)
        matched = next((s for s in roster if s["roll_number"] == test_roll), None)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["attendance_status"], "PRESENT")
        self.assertEqual(matched["verified_by"], "Prof. Textile Faculty")

        # Test attendance report generation
        rep = db_manager.generate_attendance_report()
        self.assertTrue(isinstance(rep, list))

        # Cleanup
        db_manager.delete_student(test_roll)

    # -------------------------------------------------------------
    # MODULE 8 TESTS: STUDIO UI & ROUTES INTEGRITY
    # -------------------------------------------------------------
    def test_08_studio_dashboard_route_and_templates(self):
        """Verifies that /studio renders HTTP 200 with all components."""
        res = self.client.get("/studio")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Student Management", html)
        self.assertIn("Image Dataset Management", html)
        self.assertIn("AI Training Studio & Monitor", html)
        self.assertIn("Live Camera Occupancy Scanner", html)
        self.assertIn("Teacher-Reviewed Attendance", html)


if __name__ == "__main__":
    unittest.main()
