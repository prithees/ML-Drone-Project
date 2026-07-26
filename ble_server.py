"""
BLE peripheral exposing spray control to the mobile app: connect, write
"START"/"STOP" to the control characteristic to run/stop the scan+spray
loop (spray_service.SpraySession), and read/subscribe to the status
characteristic for live state.

Setup on the Raspberry Pi (bluezero wraps BlueZ over D-Bus; dbus-python and
PyGObject are easiest installed as system packages rather than via pip):
    sudo apt install bluetooth bluez python3-dbus python3-gi
    pip install bluezero
Must run with permission to talk to BlueZ over D-Bus (typically root, or a
user granted access via a D-Bus policy file — see bluezero's docs).

These UUIDs must match BleClient in the mobile app (mobile/lib/ble_client.dart).
"""
from bluezero import adapter, peripheral
from gi.repository import GLib

from spray_service import SpraySession

DEVICE_NAME = "LeafScanDrone"

SERVICE_UUID = "d61a1a55-0c00-4b6f-9c7e-2f6b6c1a9e10"
CONTROL_CHAR_UUID = "d61a1a55-0c01-4b6f-9c7e-2f6b6c1a9e10"
STATUS_CHAR_UUID = "d61a1a55-0c02-4b6f-9c7e-2f6b6c1a9e10"

_last_status = "idle"
status_char = None  # assigned once the characteristic is created, below


def _on_control_write(value, options):
    command = bytes(value).decode("utf-8", errors="replace").strip().upper()
    if command == "START":
        session.start()
    elif command == "STOP":
        session.stop()


def _on_phone_disconnect(device):
    # Safety net: if the phone's BLE link drops mid-spray (out of range,
    # app killed, etc.), stop spraying rather than leaving the pin HIGH
    # with nothing left to send a STOP.
    print(f"Phone disconnected ({device}) — stopping spray.")
    session.stop()


def _read_status():
    return list(_last_status.encode("utf-8"))


def _push_status(status):
    status_char.set_value(list(status.encode("utf-8")))
    return False  # GLib.idle_add: run once, don't repeat


def _on_status_changed(status):
    global _last_status
    _last_status = status
    # SpraySession reports status from its own background thread, but
    # bluezero's D-Bus objects live on the GLib main loop — updates have to
    # be marshaled onto that thread instead of called directly here.
    GLib.idle_add(_push_status, status)


session = SpraySession(status_callback=_on_status_changed)

adapter_address = adapter.list_adapters()[0]
drone = peripheral.Peripheral(adapter_address, local_name=DEVICE_NAME)

drone.add_service(srv_id=1, uuid=SERVICE_UUID, primary=True)

drone.add_characteristic(
    srv_id=1, chr_id=1, uuid=CONTROL_CHAR_UUID,
    value=[], notifying=False,
    flags=["write"],
    write_callback=_on_control_write,
)

drone.add_characteristic(
    srv_id=1, chr_id=2, uuid=STATUS_CHAR_UUID,
    value=list(_last_status.encode("utf-8")), notifying=False,
    flags=["read", "notify"],
    read_callback=_read_status,
)
status_char = drone.characteristics[-1]

drone.on_disconnect = _on_phone_disconnect

if __name__ == "__main__":
    print(f"Advertising as '{DEVICE_NAME}'... (Ctrl+C to stop)")
    try:
        drone.publish()
    finally:
        session.shutdown()
