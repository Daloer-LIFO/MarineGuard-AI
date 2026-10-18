
import sqlite3
from pathlib import Path

DB_PATH = Path("data/marineguard.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                confidence REAL NOT NULL,
                latitude REAL,
                longitude REAL,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_incident(
    description,
    category,
    confidence,
    latitude=None,
    longitude=None,
    location=None,
):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO incidents
            (description, category, confidence,
             latitude, longitude, location)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                description,
                category,
                confidence,
                latitude,
                longitude,
                location,
            ),
        )

        return cursor.lastrowid


def list_incidents():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, description, category, confidence,
                   latitude, longitude, location, created_at
            FROM incidents
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]
