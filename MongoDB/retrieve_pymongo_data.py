"""
Legacy compatibility wrapper for MongoDB operations.
Delegates to unified DatabaseManager to ensure SQLite fallback and avoid deprecated pandas methods.
"""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from database_manager import db_manager


class database:
    def __init__(self):
        self.manager = db_manager
        self.db = db_manager.mongo_db
        self.client = db_manager.mongo_client

    def update(self, name_or_roll, lecture):
        # Support lecture as '1', '2', 'Hindi', or 'English'
        if str(lecture) == '1' or str(lecture).lower() == 'hindi':
            subject = 'Hindi'
        else:
            subject = 'English'

        # Find roll number or name
        students = self.manager.get_all_students()
        target_roll = None
        target_name = None

        for s in students:
            if s["roll_number"] == str(name_or_roll):
                target_roll = s["roll_number"]
                target_name = s["name"]
                break
            elif s["name"].lower() == str(name_or_roll).lower():
                target_roll = s["roll_number"]
                target_name = s["name"]
                break

        if not target_roll:
            target_roll = str(name_or_roll)
            target_name = str(name_or_roll)

        self.manager.mark_attendance(subject, target_roll, target_name)

    def export_csv(self, lecture):
        if str(lecture) == '1' or str(lecture).lower() == 'hindi':
            subject = 'Hindi'
        else:
            subject = 'English'
        return self.manager.export_csv(subject)
