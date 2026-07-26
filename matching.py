"""
Shared few-shot image matching: no trained classifier, no fixed class list.
A user enrolls 5-7 reference photos of whatever they want the drone to spray;
new photos are matched against those references by embedding similarity.

Used by both app.py (web/app enrollment/scan) and spray_service.py (drone-side scan).
"""
import os

import numpy as np
from PIL import Image

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

import db

IMG_SIZE = (224, 224)

# Similarity below this is treated as "no match, do not spray." Cosine
# similarity on MobileNetV2 ImageNet embeddings; tune against real field
# photos before relying on this for an actual spray decision. User-adjustable
# from the app's Settings screen (get/set_match_threshold), stored in the DB;
# this is only the fallback for a fresh install with no stored value yet.
DEFAULT_MATCH_THRESHOLD = 0.75

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = MobileNetV2(
            weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3)
        )
    return _embedder


def embed_image(path):
    image = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)
    return get_embedder()(array, training=False).numpy()[0]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def save_gallery(embeddings, guidance):
    conn = db.get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM reference_embedding")
            conn.executemany(
                "INSERT INTO reference_embedding (ref_index, embedding) VALUES (?, ?)",
                [
                    (i, np.asarray(e, dtype=np.float32).tobytes())
                    for i, e in enumerate(embeddings)
                ],
            )
            conn.execute("DELETE FROM target")
            conn.execute(
                "INSERT INTO target (id, guidance, num_references) VALUES (1, ?, ?)",
                (guidance, len(embeddings)),
            )
    finally:
        conn.close()


def load_gallery():
    """Returns (embeddings: np.ndarray[N, D], guidance: str) or (None, None) if no target is set."""
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT guidance FROM target WHERE id = 1").fetchone()
        if row is None:
            return None, None
        guidance = row[0]
        rows = conn.execute(
            "SELECT embedding FROM reference_embedding ORDER BY ref_index"
        ).fetchall()
        embeddings = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
        return embeddings, guidance
    finally:
        conn.close()


def best_match_score(query_embedding, reference_embeddings):
    """Highest similarity to any single reference photo (not the average)."""
    return max(cosine_similarity(query_embedding, ref) for ref in reference_embeddings)


def get_match_threshold():
    conn = db.get_connection()
    try:
        row = conn.execute("SELECT match_threshold FROM settings WHERE id = 1").fetchone()
        return row[0] if row is not None else DEFAULT_MATCH_THRESHOLD
    finally:
        conn.close()


def set_match_threshold(threshold):
    conn = db.get_connection()
    try:
        with conn:
            conn.execute(
                "INSERT INTO settings (id, match_threshold) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET match_threshold = excluded.match_threshold",
                (threshold,),
            )
    finally:
        conn.close()


def is_match(score, threshold=None):
    if threshold is None:
        threshold = get_match_threshold()
    return score >= threshold
