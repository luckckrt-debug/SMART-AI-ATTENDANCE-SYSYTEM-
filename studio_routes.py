"""
SMART AI CAM — AI Training Studio & Non-Identifying Visual Analytics Blueprint
Handles:
1. Student Management (CRUD, Search, Filter, Bulk CSV/Excel Import, CSV/Excel Export)
2. Image Dataset Management (Multi-file upload, validation, corruption, duplicates, blur/light check)
3. AI Training Studio & Monitor (Non-identifying tasks, async background worker, Chart.js metrics)
4. Non-Identifying Camera Occupancy (Real-time person bounding boxes & headcount)
5. Teacher-Reviewed Attendance (Roster, visual occupancy comparison, duplicate prevention, audit logs, reports)
"""

import os
import io
import json
import base64
import logging
import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template, Response, session, redirect, url_for, flash
import pandas as pd
import numpy as np
import cv2

from database_manager import db_manager
import studio_engine
from studio_engine import image_validator, person_detector, TrainingStudioWorker

logger = logging.getLogger("StudioRoutes")
BASE_DIR = Path(__file__).resolve().parent

studio_bp = Blueprint("studio", __name__)
studio_worker = TrainingStudioWorker(db_manager)


def get_current_user():
    user = session.get("user")
    if not user:
        return {
            "username": "admin",
            "name": "Administrator",
            "role": "ADMIN",
            "email": "admin@smartclassroom.edu"
        }
    return user


# =============================================================================
# STUDIO MAIN DASHBOARD VIEW
# =============================================================================

@studio_bp.route("/studio")
@studio_bp.route("/training-studio")
def studio_dashboard():
    user = get_current_user()
    is_admin = (user.get("role") == "ADMIN")
    active_sess = db_manager.get_active_session()

    datasets = db_manager.get_all_datasets()
    students = db_manager.get_all_students()
    training_jobs = db_manager.get_all_training_jobs(limit=10)
    model_versions = db_manager.get_model_versions()
    camera_devices = db_manager.get_camera_devices()

    # Roster for active session (or all enrolled students)
    sess_id = active_sess["session_id"] if active_sess else None
    roster = db_manager.get_teacher_attendance_roster(session_id=sess_id)

    # Class groups for dropdowns
    class_groups = db_manager.get_all_classes()

    return render_template(
        "studio.html",
        current_user=user,
        is_admin=is_admin,
        active_session=active_sess,
        datasets=datasets,
        students=students,
        training_jobs=training_jobs,
        model_versions=model_versions,
        camera_devices=camera_devices,
        roster=roster,
        class_groups=class_groups
    )


# =============================================================================
# MODULE 1: STUDENT MANAGEMENT & BULK IMPORT/EXPORT APIS
# =============================================================================

@studio_bp.route("/api/studio/students", methods=["GET"])
def api_get_students():
    query = request.args.get("query", "")
    class_name = request.args.get("class_name", "")
    section = request.args.get("section", "")
    status = request.args.get("status", "")

    students = db_manager.search_filter_students(query=query, class_name=class_name, section=section, status=status)
    return jsonify({
        "success": True,
        "total": len(students),
        "students": students
    })


@studio_bp.route("/api/studio/students/add", methods=["POST"])
def api_add_student():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        name = data.get("name", "").strip()
        roll = data.get("roll_number", "").strip()
        st_id = data.get("student_id", "").strip() or roll
        class_name = data.get("class_name", "").strip()
        section = data.get("section", "Section A").strip()
        session_val = data.get("academic_session", "2026-27").strip()
        enrollment_status = data.get("enrollment_status", "Active").strip()
        degree = data.get("degree", "B.Tech").strip()
        stream = data.get("stream", "Textile Technology").strip()
        academic_year = data.get("academic_year", "1st Year").strip()
        email = data.get("email", "").strip()
        mobile = data.get("mobile", "").strip()

        if not name or not roll:
            return jsonify({"success": False, "error": "Student Name and Roll Number are required."}), 400

        # Handle optional profile photograph upload
        photo_url = ""
        if "profile_photo" in request.files:
            file = request.files["profile_photo"]
            if file and file.filename:
                file_bytes = file.read()
                val = image_validator.inspect_image_bytes(file_bytes, file.filename)
                if val["is_valid"]:
                    dest_dir = BASE_DIR / "static" / "profiles"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"{roll}_{Path(file.filename).name}"
                    (dest_dir / fname).write_bytes(file_bytes)
                    photo_url = f"/static/profiles/{fname}"

        st = db_manager.add_student_record(
            name=name,
            roll_number=roll,
            student_id=st_id,
            class_name=class_name or f"{degree} {academic_year}",
            section=section,
            academic_session=session_val,
            profile_photo=photo_url,
            enrollment_status=enrollment_status,
            degree=degree,
            stream=stream,
            academic_year=academic_year,
            email=email,
            mobile=mobile
        )
        return jsonify({"success": True, "student": st, "message": f"Student '{name}' registered successfully."})
    except Exception as e:
        logger.error(f"[Add Student Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/students/edit", methods=["POST"])
def api_edit_student():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        roll = data.get("roll_number", "").strip()
        if not roll:
            return jsonify({"success": False, "error": "Roll number is required to edit."}), 400

        st = db_manager.edit_student_record(
            roll_number=roll,
            name=data.get("name"),
            student_id=data.get("student_id"),
            class_name=data.get("class_name"),
            section=data.get("section"),
            academic_session=data.get("academic_session"),
            enrollment_status=data.get("enrollment_status"),
            degree=data.get("degree"),
            stream=data.get("stream"),
            academic_year=data.get("academic_year"),
            email=data.get("email"),
            mobile=data.get("mobile")
        )
        if not st:
            return jsonify({"success": False, "error": f"Student with roll number '{roll}' not found."}), 404

        return jsonify({"success": True, "student": st, "message": "Student profile updated successfully."})
    except Exception as e:
        logger.error(f"[Edit Student Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/students/delete", methods=["POST"])
def api_delete_student():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        roll = data.get("roll_number", "").strip()
        if not roll:
            return jsonify({"success": False, "error": "Roll number required for deletion."}), 400

        success = db_manager.delete_student(roll)
        if success:
            return jsonify({"success": True, "message": f"Student '{roll}' and associated records purged."})
        return jsonify({"success": False, "error": f"Student '{roll}' could not be deleted."}), 400
    except Exception as e:
        logger.error(f"[Delete Student Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/students/import-bulk", methods=["POST"])
def api_bulk_import_students():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded. Please upload a CSV or Excel (.xlsx) file."}), 400

        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            return jsonify({"success": False, "error": "Empty filename."}), 400

        ext = Path(uploaded_file.filename).suffix.lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            return jsonify({"success": False, "error": f"Invalid format '{ext}'. Only .csv and .xlsx supported."}), 400

        # Read into pandas DataFrame
        if ext == ".csv":
            df = pd.read_csv(uploaded_file.stream, dtype=str).fillna("")
        else:
            df = pd.read_excel(uploaded_file.stream, dtype=str).fillna("")

        records = df.to_dict(orient="records")
        res = db_manager.bulk_import_students(records)
        return jsonify(res)
    except Exception as e:
        logger.error(f"[Bulk Import Error] {e}")
        return jsonify({"success": False, "error": f"Failed to parse file: {e}"}), 500


@studio_bp.route("/api/studio/students/export-csv")
def api_export_students_csv():
    students = db_manager.get_all_students()
    df = pd.DataFrame(students)
    columns_order = [
        "name", "roll_number", "student_id", "degree", "stream",
        "academic_year", "section", "academic_session", "enrollment_status",
        "email", "mobile", "created_at"
    ]
    avail_cols = [c for c in columns_order if c in df.columns]
    df = df[avail_cols] if not df.empty else pd.DataFrame(columns=columns_order)

    csv_data = df.to_csv(index=False)
    filename = f"Students_Directory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@studio_bp.route("/api/studio/students/export-excel")
def api_export_students_excel():
    students = db_manager.get_all_students()
    df = pd.DataFrame(students)
    columns_order = [
        "name", "roll_number", "student_id", "degree", "stream",
        "academic_year", "section", "academic_session", "enrollment_status",
        "email", "mobile", "created_at"
    ]
    avail_cols = [c for c in columns_order if c in df.columns]
    df = df[avail_cols] if not df.empty else pd.DataFrame(columns=columns_order)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Students", index=False)
    output.seek(0)

    filename = f"Students_Directory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        output.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# =============================================================================
# MODULE 2: IMAGE DATASET MANAGEMENT APIS
# =============================================================================

@studio_bp.route("/api/studio/datasets", methods=["GET", "POST"])
def api_datasets_handler():
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or request.form.to_dict()
            name = data.get("name", "").strip()
            desc = data.get("description", "").strip()
            task_type = data.get("task_type", "person_detection").strip()
            train_split = float(data.get("train_split", 0.70))
            val_split = float(data.get("val_split", 0.15))
            test_split = float(data.get("test_split", 0.15))

            if not name:
                return jsonify({"success": False, "error": "Dataset name is required."}), 400

            ds = db_manager.create_dataset(
                name=name,
                description=desc,
                task_type=task_type,
                train_split=train_split,
                val_split=val_split,
                test_split=test_split
            )
            return jsonify({"success": True, "dataset": ds, "message": f"Dataset '{name}' created."})
        except Exception as e:
            logger.error(f"[Create Dataset Error] {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    datasets = db_manager.get_all_datasets()
    return jsonify({"success": True, "datasets": datasets})


@studio_bp.route("/api/studio/datasets/<dataset_id>/delete", methods=["POST"])
def api_delete_dataset(dataset_id):
    try:
        success = db_manager.delete_dataset(dataset_id)
        return jsonify({"success": success, "message": "Dataset and files deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/datasets/<dataset_id>/upload", methods=["POST"])
def api_upload_dataset_images(dataset_id):
    """
    Accepts multiple images.
    Performs corruption check, dimension check, duplicate check (SHA256), blur, and lighting analysis.
    Assigns split category based on dataset ratios.
    """
    try:
        ds = db_manager.get_dataset(dataset_id)
        if not ds:
            return jsonify({"success": False, "error": f"Dataset '{dataset_id}' not found."}), 404

        files = request.files.getlist("images") or request.files.getlist("file")
        if not files or len(files) == 0:
            return jsonify({"success": False, "error": "No image files provided for upload."}), 400

        target_dir = BASE_DIR / "static" / "datasets" / dataset_id
        target_dir.mkdir(parents=True, exist_ok=True)

        uploaded = []
        duplicates = 0
        corrupted = 0
        skipped = 0

        # Ratios
        tr_split = ds.get("train_split", 0.70)
        val_split = ds.get("val_split", 0.15)

        for f in files:
            if not f or not f.filename:
                continue
            raw_bytes = f.read()
            if len(raw_bytes) == 0:
                continue

            # Run validator
            val = image_validator.inspect_image_bytes(raw_bytes, f.filename)
            if not val["is_valid"]:
                corrupted += 1
                continue

            # Check duplicate in dataset
            if db_manager.check_dataset_duplicate(dataset_id, val["file_hash"]):
                duplicates += 1
                continue

            # Assign split category
            r = np.random.rand()
            if r < tr_split:
                split_cat = "train"
            elif r < (tr_split + val_split):
                split_cat = "val"
            else:
                split_cat = "test"

            # Save file physically
            safe_fname = f"{val['file_hash'][:10]}_{Path(f.filename).name}"
            dest_file = target_dir / safe_fname
            dest_file.write_bytes(raw_bytes)
            web_path = f"/static/datasets/{dataset_id}/{safe_fname}"

            # Save to DB
            file_id = db_manager.add_dataset_file(
                dataset_id=dataset_id,
                file_path=web_path,
                file_name=safe_fname,
                file_hash=val["file_hash"],
                file_size=val["file_size"],
                width=val["width"],
                height=val["height"],
                quality_score=val["quality_score"],
                is_corrupted=val["is_corrupted"],
                is_duplicate=0,
                is_low_light=val["is_low_light"],
                is_blurred=val["is_blurred"],
                split_category=split_cat
            )
            val["file_id"] = file_id
            val["file_path"] = web_path
            val["split_category"] = split_cat
            uploaded.append(val)

        return jsonify({
            "success": True,
            "uploaded_count": len(uploaded),
            "duplicates_count": duplicates,
            "corrupted_count": corrupted,
            "total_files": len(uploaded),
            "files": uploaded,
            "message": f"Successfully processed {len(uploaded)} images ({duplicates} duplicates skipped, {corrupted} corrupted)."
        })

    except Exception as e:
        logger.error(f"[Dataset Upload Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/datasets/<dataset_id>/files", methods=["GET"])
def api_get_dataset_files(dataset_id):
    split = request.args.get("split")
    files = db_manager.get_dataset_files(dataset_id, split=split)
    return jsonify({"success": True, "total": len(files), "files": files})


@studio_bp.route("/api/studio/datasets/<dataset_id>/files/<file_id>/delete", methods=["POST"])
def api_delete_dataset_file(dataset_id, file_id):
    try:
        success = db_manager.delete_dataset_file(file_id)
        return jsonify({"success": success, "message": "Image deleted."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# MODULE 3 & 4: AI TRAINING STUDIO & MONITOR APIS
# =============================================================================

@studio_bp.route("/api/studio/train/start", methods=["POST"])
def api_start_training_job():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        job_name = data.get("job_name", "").strip() or f"Training-{datetime.datetime.now().strftime('%H%M%S')}"
        dataset_id = data.get("dataset_id", "").strip()
        task_type = data.get("task_type", "person_detection").strip()
        model_arch = data.get("model_architecture", "MobileNet-PersonDetector").strip()
        epochs = int(data.get("epochs", 10))
        batch_size = int(data.get("batch_size", 16))
        learning_rate = float(data.get("learning_rate", 0.001))
        optimizer = data.get("optimizer", "adam").strip()

        if not dataset_id:
            return jsonify({"success": False, "error": "Please select a dataset to train."}), 400

        # Prevent concurrent conflicting jobs
        active_jobs = [j for j in db_manager.get_all_training_jobs(limit=5) if j.get("status") == "running"]
        if active_jobs:
            return jsonify({
                "success": False,
                "error": f"A training job ('{active_jobs[0]['job_name']}') is currently in progress. Please wait or stop it."
            }), 409

        job = db_manager.create_training_job(
            job_name=job_name,
            dataset_id=dataset_id,
            task_type=task_type,
            model_architecture=model_arch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer
        )

        # Launch background worker
        studio_worker.launch_training(
            job_id=job["id"],
            dataset_id=dataset_id,
            task_type=task_type,
            architecture=model_arch,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer
        )

        return jsonify({
            "success": True,
            "job": job,
            "message": f"Training job '{job_name}' started successfully in background."
        })

    except Exception as e:
        logger.error(f"[Start Training Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/train/stop/<job_id>", methods=["POST"])
def api_stop_training_job(job_id):
    try:
        success = studio_worker.stop_job(job_id)
        return jsonify({"success": success, "message": "Stop signal delivered to training worker."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/train/status/<job_id>", methods=["GET"])
def api_get_training_job_status(job_id):
    job = db_manager.get_training_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "Job not found"}), 404
    logs = db_manager.get_training_logs(job_id, limit=30)
    return jsonify({"success": True, "job": job, "recent_logs": logs})


@studio_bp.route("/api/studio/train/logs/<job_id>", methods=["GET"])
def api_get_training_job_logs(job_id):
    limit = int(request.args.get("limit", 100))
    logs = db_manager.get_training_logs(job_id, limit=limit)
    return jsonify({"success": True, "logs": logs})


@studio_bp.route("/api/studio/models", methods=["GET"])
def api_get_model_versions():
    task = request.args.get("task_type")
    versions = db_manager.get_model_versions(task_type=task)
    return jsonify({"success": True, "models": versions})


@studio_bp.route("/api/studio/models/<version_id>/deploy", methods=["POST"])
def api_deploy_model_version(version_id):
    try:
        success = db_manager.set_active_model_version(version_id)
        if success:
            return jsonify({"success": True, "message": f"Model '{version_id}' is now ACTIVE for live inference."})
        return jsonify({"success": False, "error": "Model version not found."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# MODULE 5: NON-IDENTIFYING CAMERA OCCUPANCY INFERENCE API
# =============================================================================

@studio_bp.route("/api/studio/camera/process-frame", methods=["POST"])
def api_process_studio_camera_frame():
    """
    Receives camera frame base64.
    Runs non-identifying person detection and occupancy counting.
    Returns annotated frame and real-time room analytics.
    Strictly preserves face privacy.
    """
    try:
        data = request.get_json(silent=True) or {}
        frame_data = data.get("frame", "")
        conf_threshold = float(data.get("confidence", 0.50))
        cam_id = data.get("camera_id", "CAM_01")

        if not frame_data:
            return jsonify({"success": False, "error": "No image frame provided"}), 400

        # Decode base64
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]
        img_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Failed to decode frame"}), 400

        # Process with NonIdentifyingPersonDetector
        analysis = person_detector.analyze_frame(frame, conf_threshold=conf_threshold)

        # Encode annotated image
        annotated = analysis["annotated_frame"]
        annotated_b64 = ""
        if annotated is not None:
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        # Update camera device status in DB
        db_manager.update_camera_status(cam_id, status="ONLINE", fps=18.5, latency_ms=42)

        return jsonify({
            "success": True,
            "occupancy_count": analysis["occupancy_count"],
            "persons": analysis["persons"],
            "quality": analysis["quality"],
            "scene": analysis["scene"],
            "annotated_frame": annotated_b64
        })

    except Exception as e:
        logger.error(f"[Studio Camera Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# MODULE 6: TEACHER-REVIEWED ATTENDANCE APIS
# =============================================================================

@studio_bp.route("/api/studio/attendance/roster", methods=["GET"])
def api_get_attendance_roster():
    session_id = request.args.get("session_id")
    section = request.args.get("section")
    roster = db_manager.get_teacher_attendance_roster(session_id=session_id, section=section)
    present_cnt = sum(1 for s in roster if s.get("attendance_status") == "PRESENT")
    absent_cnt = sum(1 for s in roster if s.get("attendance_status") == "ABSENT")
    late_cnt = sum(1 for s in roster if s.get("attendance_status") == "LATE")

    return jsonify({
        "success": True,
        "total_enrolled": len(roster),
        "present_count": present_cnt,
        "absent_count": absent_cnt,
        "late_count": late_cnt,
        "roster": roster
    })


@studio_bp.route("/api/studio/attendance/verify", methods=["POST"])
def api_verify_attendance():
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        session_id = data.get("session_id", "").strip()
        student_id = data.get("student_id", "").strip()
        status = data.get("status", "PRESENT").strip().upper()
        audit_notes = data.get("audit_notes", "Verified via Teacher Attendance Studio").strip()
        user = get_current_user()
        verified_by = user.get("name", "Teacher")

        if not session_id or not student_id:
            return jsonify({"success": False, "error": "session_id and student_id are required."}), 400

        db_manager.teacher_verify_attendance(
            session_id=session_id,
            student_id=student_id,
            status=status,
            verified_by=verified_by,
            audit_notes=audit_notes
        )

        return jsonify({
            "success": True,
            "message": f"Student '{student_id}' marked as {status} by {verified_by}."
        })
    except Exception as e:
        logger.error(f"[Attendance Verify Error] {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@studio_bp.route("/api/studio/attendance/report")
def api_download_attendance_report():
    date_str = request.args.get("date")
    month_str = request.args.get("month")
    as_csv = request.args.get("format", "csv").lower() == "csv"

    records = db_manager.generate_attendance_report(date_str=date_str, month_str=month_str)
    if not as_csv:
        return jsonify({"success": True, "total": len(records), "records": records})

    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["attendance_id", "session_id", "student_name", "roll_number", "status", "first_check_in", "verified_by", "audit_notes"])

    csv_data = df.to_csv(index=False)
    fname = f"Attendance_Report_{date_str or month_str or 'recent'}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )
