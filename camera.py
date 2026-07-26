"""
Live capture from the Raspberry Pi Camera Module (CSI), used by
spray_service.py instead of reading pre-existing numbered image files.

Captures straight to a JPEG file and hands the path to matching.embed_image,
reusing the same PIL-based load path already used for enrollment/web
uploads, instead of dealing with picamera2's raw in-memory array formats.
"""
import os

from picamera2 import Picamera2

CAPTURE_PATH = "uploads/camera_frame.jpg"
CAPTURE_SIZE = (1024, 768)


class Camera:
    def __init__(self, capture_size=CAPTURE_SIZE):
        self._picam2 = Picamera2()
        config = self._picam2.create_still_configuration(main={"size": capture_size})
        self._picam2.configure(config)
        self._picam2.start()

    def capture(self, path=CAPTURE_PATH):
        """Captures a frame to `path` and returns the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._picam2.capture_file(path)
        return path

    def close(self):
        self._picam2.stop()
        self._picam2.close()
