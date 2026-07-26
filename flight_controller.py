"""
Read-only MAVLink telemetry from the flight controller: position and
groundspeed. Used by spray_service.py instead of the standalone serial GPS
(gps.py) when a real flight controller is connected — this reads the same
GPS/EKF-derived position the autopilot navigates with, so the spray-area
geofence check matches what the aircraft actually believes its position is.

This is intentionally read-only: it never sends commands to the flight
controller. Position/groundspeed only.

Typical connection strings:
  - "/dev/serial0" or "/dev/ttyAMA0" at 57600 baud: Pi UART wired directly
    to a telemetry port on the flight controller (most common setup).
  - "udp:127.0.0.1:14550": a MAVLink stream forwarded over UDP (e.g. via
    MAVProxy or the flight controller's own companion-computer link).
"""
from pymavlink import mavutil

FC_CONNECTION = "/dev/serial0"
FC_BAUD = 57600
HEARTBEAT_TIMEOUT_SECONDS = 5


class FlightControllerLink:
    def __init__(self, connection=FC_CONNECTION, baud=FC_BAUD):
        self._master = mavutil.mavlink_connection(connection, baud=baud)
        if self._master.wait_heartbeat(timeout=HEARTBEAT_TIMEOUT_SECONDS) is None:
            self._master.close()
            raise TimeoutError(
                f"No MAVLink heartbeat from flight controller on {connection} "
                f"within {HEARTBEAT_TIMEOUT_SECONDS}s."
            )
        self._last_fix = None
        self._last_groundspeed = None

    def _drain(self):
        """Reads all pending MAVLink messages (non-blocking), updating the
        latest known position/groundspeed. Call before each read_*()."""
        while True:
            msg = self._master.recv_match(type=["GLOBAL_POSITION_INT", "VFR_HUD"], blocking=False)
            if msg is None:
                break
            msg_type = msg.get_type()
            if msg_type == "GLOBAL_POSITION_INT":
                self._last_fix = (msg.lat / 1e7, msg.lon / 1e7)
            elif msg_type == "VFR_HUD":
                self._last_groundspeed = msg.groundspeed

    def read_fix(self):
        """Returns the latest (lat, lng) as floats, or None if no position
        report has been received yet. Same shape as gps.GpsReader.read_fix()
        so spray_service.py can use either interchangeably."""
        self._drain()
        return self._last_fix

    def read_groundspeed(self):
        """Returns the latest groundspeed in m/s, or None if not seen yet."""
        self._drain()
        return self._last_groundspeed

    def close(self):
        self._master.close()
