"""
Scan/spray event history: appended to on every scan (app.py, via the web/API
scan endpoints) and every spray-loop iteration (spray_service.SpraySession),
so the mobile app can show what happened and where.

SQLite (db.py) instead of a flat file: real indexed queries/LIMIT instead of
loading everything into memory, and safe concurrent appends from app.py and
spray_service.py's background thread without a full-file read-modify-write.
"""
import time

import db

MAX_ENTRIES = 500


def log_event(matched, score, sprayed, lat=None, lng=None, source="api"):
    conn = db.get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO history (timestamp, source, matched, score, sprayed, lat, lng) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), source, int(matched), score, int(sprayed), lat, lng),
            )
            conn.execute(
                "DELETE FROM history WHERE id NOT IN "
                "(SELECT id FROM history ORDER BY id DESC LIMIT ?)",
                (MAX_ENTRIES,),
            )
    finally:
        conn.close()


def load_history(limit=None):
    """Returns entries most-recent-first, optionally capped to `limit`."""
    conn = db.get_connection()
    try:
        query = "SELECT timestamp, source, matched, score, sprayed, lat, lng FROM history ORDER BY id DESC"
        params = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        rows = conn.execute(query, params).fetchall()
        return [
            {
                "timestamp": timestamp,
                "source": source,
                "matched": bool(matched),
                "score": score,
                "sprayed": bool(sprayed),
                "lat": lat,
                "lng": lng,
            }
            for timestamp, source, matched, score, sprayed, lat, lng in rows
        ]
    finally:
        conn.close()
