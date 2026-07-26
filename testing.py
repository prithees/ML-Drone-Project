"""
Manual/local runner for the spray loop, without needing a BLE connection —
useful for testing on the bench. For normal operation, the mobile app
starts/stops this same loop over BLE via ble_server.py.
"""
import time

import keyboard

from spray_service import SpraySession

session = SpraySession(status_callback=print)
session.start()

print("Spraying loop started. Press Enter to stop.")
try:
    while not keyboard.is_pressed('enter'):
        time.sleep(0.1)
    print("Enter pressed, stopping.")
finally:
    session.shutdown()
