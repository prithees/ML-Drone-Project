"""
Spray-area geofence: a polygon boundary within which the drone is allowed to
spray, configured from the mobile app's map screen.

Used by app.py (save/load the polygon) and spray_service.py (check the
drone's current GPS position against it before spraying).
"""
import db

MIN_POLYGON_POINTS = 3


def save_area(points):
    """points: list of (lat, lng) tuples."""
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM spray_area_point")
            conn.executemany(
                "INSERT INTO spray_area_point (seq, lat, lng) VALUES (?, ?, ?)",
                [(i, lat, lng) for i, (lat, lng) in enumerate(points)],
            )
    finally:
        conn.close()


def load_area():
    """Returns a list of (lat, lng) tuples, or None if no area is set."""
    conn = db.get_connection()
    try:
        rows = conn.execute("SELECT lat, lng FROM spray_area_point ORDER BY seq").fetchall()
        return [(lat, lng) for lat, lng in rows] if rows else None
    finally:
        conn.close()


def point_in_polygon(lat, lng, polygon):
    """Ray-casting point-in-polygon test. polygon: list of (lat, lng) tuples."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        lat_i, lng_i = polygon[i]
        lat_j, lng_j = polygon[j]
        intersects = ((lng_i > lng) != (lng_j > lng)) and (
            lat < (lat_j - lat_i) * (lng - lng_i) / (lng_j - lng_i) + lat_i
        )
        if intersects:
            inside = not inside
        j = i
    return inside
