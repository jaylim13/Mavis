import sqlite3
import uuid
from datetime import datetime

DB_PATH = "mavis_memory.db"


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT NOT NULL DEFAULT '',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    return conn


def start_session(conn):
    session_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
    conn.commit()
    return session_id


def load_summary(conn):
    row = conn.execute("SELECT content FROM summary WHERE id = 1").fetchone()
    return row[0] if row else ""


def save_summary(conn, summary_text):
    conn.execute(
        "INSERT INTO summary (id, content) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET content = excluded.content, "
        "updated_at = CURRENT_TIMESTAMP",
        (summary_text,)
    )
    conn.commit()


def log_turn(conn, session_id, role, text):
    conn.execute(
        "INSERT INTO turns (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, text)
    )
    conn.commit()