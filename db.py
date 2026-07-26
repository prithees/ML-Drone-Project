"""
Shared SQLite storage for the target gallery, spray area, and scan history —
used by matching.py, spray_area.py, and history.py so they all read/write
one file instead of scattering separate JSON/npz files across uploads/.

app.py and ble_server.py/spray_service.py run as separate processes but both
touch this file, so connections use WAL mode: readers don't block on a
writer and vice versa, which matters since a scan can come in over HTTP at
the same time the drone loop is writing a history row.
"""
import os
import sqlite3

DB_PATH = "uploads/drone.db"


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    _init_schema(conn)
    return conn


def _init_schema(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS target (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            guidance TEXT NOT NULL,
            num_references INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reference_embedding (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ref_index INTEGER NOT NULL,
            embedding BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS spray_area_point (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seq INTEGER NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            source TEXT NOT NULL,
            matched INTEGER NOT NULL,
            score REAL NOT NULL,
            sprayed INTEGER NOT NULL,
            lat REAL,
            lng REAL
        );
        CREATE INDEX IF NOT EXISTS idx_history_id_desc ON history (id DESC);
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            match_threshold REAL NOT NULL
        );
    """)
    conn.commit()
