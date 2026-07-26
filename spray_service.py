"""
Spray control service: owns the scan+spray loop as a start/stoppable
background thread, so it can be driven by an external trigger (the BLE
control characteristic in ble_server.py) instead of running once as a
top-level script.

Scans live camera frames (camera.Camera) rather than pre-existing numbered
image files, and — when a spray area is configured — checks position against
it using the flight controller's own GPS/EKF over MAVLink
(flight_controller.FlightControllerLink) rather than a standalone GPS module.
"""
import threading
import time

import RPi.GPIO as GPIO

import camera
import flight_controller
import history
import matching
import spray_area

LED_PIN = 14
SPRAY_DURATION_SECONDS = 2


class SpraySession:
    """Runs the scan+spray loop on a background thread until stopped.

    GPIO is set up once for the lifetime of the session so start()/stop()
    can be called repeatedly; call shutdown() once when the process exits
    to release the GPIO channel.
    """

    def __init__(self, status_callback=None):
        self._status_callback = status_callback or (lambda status: None)
        self._thread = None
        self._stop_event = threading.Event()

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LED_PIN, GPIO.OUT)
        GPIO.output(LED_PIN, GPIO.LOW)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SPRAY_DURATION_SECONDS + 5)
        GPIO.output(LED_PIN, GPIO.LOW)
        self._status_callback("idle")

    def shutdown(self):
        self.stop()
        GPIO.cleanup()

    def _run(self):
        reference_embeddings, guidance = matching.load_gallery()
        if reference_embeddings is None:
            self._status_callback("error: no target set")
            return

        spray_boundary = spray_area.load_area()
        fc_link = None
        if spray_boundary is not None:
            try:
                fc_link = flight_controller.FlightControllerLink()
            except Exception as e:
                self._status_callback(f"error: no flight controller telemetry ({e})")
                return

        try:
            cam = camera.Camera()
        except Exception as e:
            self._status_callback(f"error: camera unavailable ({e})")
            if fc_link is not None:
                fc_link.close()
            return

        self._status_callback("scanning")
        try:
            frame_num = 0
            while not self._stop_event.is_set():
                frame_num += 1
                try:
                    frame_path = cam.capture()
                    query_embedding = matching.embed_image(frame_path)
                    score = matching.best_match_score(query_embedding, reference_embeddings)
                    matched = matching.is_match(score)

                    fix = None
                    if matched and spray_boundary is not None:
                        fix = fc_link.read_fix()
                        if fix is None or not spray_area.point_in_polygon(fix[0], fix[1], spray_boundary):
                            matched = False

                    if matched:
                        self._status_callback(f"spraying: frame {frame_num}")
                        GPIO.output(LED_PIN, GPIO.HIGH)
                        time.sleep(SPRAY_DURATION_SECONDS)
                        GPIO.output(LED_PIN, GPIO.LOW)
                    else:
                        self._status_callback(f"scanning: frame {frame_num} no match")

                    lat, lng = fix if fix else (None, None)
                    history.log_event(matched, score, sprayed=matched, lat=lat, lng=lng, source="drone")
                except Exception as e:
                    self._status_callback(f"error: {e}")
        finally:
            GPIO.output(LED_PIN, GPIO.LOW)
            cam.close()
            if fc_link is not None:
                fc_link.close()
            if not self._stop_event.is_set():
                self._status_callback("idle: finished")
