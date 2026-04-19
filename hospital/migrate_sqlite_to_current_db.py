r"""
Copy data from a local SQLite backup into the database currently configured
for the app.

Typical Render workflow:
  1. Set DATABASE_URL in the Render web service to the Render Postgres URL.
  2. Run this script locally with your old SQLite file as the source.

Examples:
  python migrate_sqlite_to_current_db.py
  python migrate_sqlite_to_current_db.py --source .\instance\hospital_kiosk.db
  python migrate_sqlite_to_current_db.py --replace

The script preserves the important app data:
  - users
  - departments
  - patients
  - emergency logs
  - chat messages
  - captured faces
  - face recognition model settings
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys
from typing import Any, Iterable

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, Time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hospital_kiosk_web import (  # noqa: E402
    CapturedFace,
    ChatMessage,
    Department,
    EmergencyLog,
    FaceRecognitionModel,
    Patient,
    User,
    app,
    db,
)


TABLE_DELETE_ORDER = [
    ChatMessage,
    CapturedFace,
    EmergencyLog,
    Patient,
    FaceRecognitionModel,
    User,
    Department,
]


def _parse_datetime(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value

    raw = str(value).strip().replace(" ", "T")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return dt.datetime.strptime(str(value), fmt)
            except ValueError:
                pass
    raise ValueError(f"Could not parse datetime value: {value!r}")


def _parse_date(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return dt.date.fromisoformat(str(value).strip()[:10])


def _parse_time(value: Any):
    if value is None or value == "":
        return None
    if isinstance(value, dt.time):
        return value

    raw = str(value).strip()
    if " " in raw:
        raw = raw.rsplit(" ", 1)[-1]
    if raw.endswith("Z"):
        raw = raw[:-1]
    return dt.time.fromisoformat(raw)


def _coerce_value(column, value):
    if value is None:
        return None

    if isinstance(column.type, DateTime):
        return _parse_datetime(value)
    if isinstance(column.type, Date):
        return _parse_date(value)
    if isinstance(column.type, Time):
        return _parse_time(value)
    if isinstance(column.type, Boolean):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y"}
        return bool(value)
    if isinstance(column.type, Integer):
        return int(value)
    if isinstance(column.type, Float):
        return float(value)
    return value


def _resolve_source_path(explicit_source: str | None) -> str:
    candidates = []
    if explicit_source:
        candidates.append(explicit_source)

    for env_name in ("SOURCE_SQLITE_PATH", "HOSPITAL_KIOSK_SOURCE_DB"):
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(env_value)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.join(script_dir, "instance", "hospital_kiosk.db"),
            os.path.abspath(os.path.join(script_dir, "..", "instance", "hospital_kiosk.db")),
        ]
    )

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "Could not find a source SQLite database. "
        "Pass --source or set SOURCE_SQLITE_PATH."
    )


def _fetch_rows(source_conn: sqlite3.Connection, table_name: str):
    cursor = source_conn.execute(f'SELECT * FROM "{table_name}"')
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description or []]
    return columns, rows


def _build_field_map(model, source_columns: Iterable[str], row: sqlite3.Row):
    source_column_set = set(source_columns)
    data = {}
    for column in model.__table__.columns:
        if column.name == "id":
            continue
        if column.name not in source_column_set:
            continue
        data[column.name] = _coerce_value(column, row[column.name])
    return data


def _clear_target_tables():
    for model in TABLE_DELETE_ORDER:
        deleted = model.query.delete(synchronize_session=False)
        print(f"[*] Cleared {deleted} rows from {model.__tablename__}")
    db.session.commit()


def _sync_users(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "user")
    if replace:
        User.query.delete(synchronize_session=False)
        db.session.commit()

    user_id_map = {}
    created = updated = 0

    for row in rows:
        username = row["username"]
        user = User.query.filter_by(username=username).one_or_none()
        if user is None:
            user = User(username=username, password_hash="")
            db.session.add(user)
            created += 1
        else:
            updated += 1

        fields = _build_field_map(User, source_columns, row)
        for field_name, field_value in fields.items():
            setattr(user, field_name, field_value)

        db.session.flush()
        user_id_map[row["id"]] = user.id

    return user_id_map, created, updated


def _sync_departments(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "department")
    if replace:
        Department.query.delete(synchronize_session=False)
        db.session.commit()

    created = updated = 0
    for row in rows:
        dept = Department.query.filter_by(dept_id=row["dept_id"]).one_or_none()
        if dept is None:
            dept = Department(dept_id=row["dept_id"], name="")
            db.session.add(dept)
            created += 1
        else:
            updated += 1

        fields = _build_field_map(Department, source_columns, row)
        for field_name, field_value in fields.items():
            setattr(dept, field_name, field_value)

    return created, updated


def _sync_face_model(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "face_recognition_model")
    if not rows:
        return 0, 0

    if replace:
        FaceRecognitionModel.query.delete(synchronize_session=False)
        db.session.commit()

    created = updated = 0
    row = rows[0]
    model = FaceRecognitionModel.query.first()
    if model is None:
        model = FaceRecognitionModel()
        db.session.add(model)
        created = 1
    else:
        updated = 1

    fields = _build_field_map(FaceRecognitionModel, source_columns, row)
    for field_name, field_value in fields.items():
        setattr(model, field_name, field_value)

    return created, updated


def _sync_patients(source_conn, user_id_map, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "patient")
    if replace:
        Patient.query.delete(synchronize_session=False)
        db.session.commit()

    created = updated = 0
    for row in rows:
        patient = Patient.query.filter_by(patient_id=row["patient_id"]).one_or_none()
        if patient is None:
            patient = Patient(
                patient_id=row["patient_id"],
                name=row["name"],
                age=row["age"],
                gender=row["gender"],
                phone=row["phone"],
                department=row["department"],
                consultation_type=row["consultation_type"],
            )
            db.session.add(patient)
            created += 1
        else:
            updated += 1

        fields = _build_field_map(Patient, source_columns, row)
        for field_name, field_value in fields.items():
            if field_name == "user_id":
                continue
            setattr(patient, field_name, field_value)

        source_user_id = row["user_id"] if "user_id" in source_columns else None
        patient.user_id = user_id_map.get(source_user_id) if source_user_id is not None else None

        if patient.consultation_date is None and patient.check_in_time:
            patient.consultation_date = patient.check_in_time.date()
        if patient.consultation_time is None and patient.check_in_time:
            patient.consultation_time = patient.check_in_time.time().replace(microsecond=0)

    return created, updated


def _sync_emergency_logs(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "emergency_log")
    if replace:
        EmergencyLog.query.delete(synchronize_session=False)
        db.session.commit()

    created = updated = 0
    for row in rows:
        entry = EmergencyLog.query.filter_by(emergency_id=row["emergency_id"]).one_or_none()
        if entry is None:
            entry = EmergencyLog(emergency_id=row["emergency_id"])
            db.session.add(entry)
            created += 1
        else:
            updated += 1

        fields = _build_field_map(EmergencyLog, source_columns, row)
        for field_name, field_value in fields.items():
            setattr(entry, field_name, field_value)

    return created, updated


def _sync_chat_messages(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "chat_message")
    if replace:
        ChatMessage.query.delete(synchronize_session=False)
        db.session.commit()

    created = 0
    for row in rows:
        message = ChatMessage()
        fields = _build_field_map(ChatMessage, source_columns, row)
        for field_name, field_value in fields.items():
            setattr(message, field_name, field_value)
        db.session.add(message)
        created += 1

    return created


def _sync_captured_faces(source_conn, replace=False):
    source_columns, rows = _fetch_rows(source_conn, "captured_face")
    if replace:
        CapturedFace.query.delete(synchronize_session=False)
        db.session.commit()

    created = 0
    for row in rows:
        face = CapturedFace()
        fields = _build_field_map(CapturedFace, source_columns, row)
        for field_name, field_value in fields.items():
            setattr(face, field_name, field_value)
        db.session.add(face)
        created += 1

    return created


def migrate(source_path: str, replace: bool = False):
    source_path = os.path.abspath(source_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source database not found: {source_path}")

    print(f"[*] Source SQLite: {source_path}")
    print(f"[*] Target DB: {app.config['SQLALCHEMY_DATABASE_URI']}")

    source_conn = sqlite3.connect(source_path)
    source_conn.row_factory = sqlite3.Row

    try:
        with app.app_context():
            db.create_all()
            if replace:
                print("[*] Clearing target tables before import...")
                _clear_target_tables()

            user_id_map, users_created, users_updated = _sync_users(source_conn, replace=replace)
            dept_created, dept_updated = _sync_departments(source_conn, replace=replace)
            model_created, model_updated = _sync_face_model(source_conn, replace=replace)
            patient_created, patient_updated = _sync_patients(source_conn, user_id_map, replace=replace)
            emergency_created, emergency_updated = _sync_emergency_logs(source_conn, replace=replace)
            chat_created = _sync_chat_messages(source_conn, replace=replace)
            face_created = _sync_captured_faces(source_conn, replace=replace)

            db.session.commit()

            print("\n[OK] Migration complete")
            print(f"  Users: {users_created} created, {users_updated} updated")
            print(f"  Departments: {dept_created} created, {dept_updated} updated")
            print(f"  Face model rows: {model_created} created, {model_updated} updated")
            print(f"  Patients: {patient_created} created, {patient_updated} updated")
            print(f"  Emergency logs: {emergency_created} created, {emergency_updated} updated")
            print(f"  Chat messages: {chat_created} created")
            print(f"  Captured faces: {face_created} created")
    except Exception:
        db.session.rollback()
        raise
    finally:
        source_conn.close()


def main():
    parser = argparse.ArgumentParser(description="Copy SQLite data into the current app database.")
    parser.add_argument(
        "--source",
        help="Path to the old SQLite database file. Defaults to the bundled local hospital_kiosk.db.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear the target tables before importing the source data.",
    )
    args = parser.parse_args()

    source_path = _resolve_source_path(args.source)
    migrate(source_path, replace=args.replace)


if __name__ == "__main__":
    main()
