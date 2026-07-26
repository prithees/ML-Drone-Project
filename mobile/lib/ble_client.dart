import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

/// These UUIDs must match SERVICE_UUID / CONTROL_CHAR_UUID / STATUS_CHAR_UUID
/// in ble_server.py on the drone.
final Guid sprayServiceUuid = Guid('d61a1a55-0c00-4b6f-9c7e-2f6b6c1a9e10');
final Guid sprayControlCharUuid = Guid('d61a1a55-0c01-4b6f-9c7e-2f6b6c1a9e10');
final Guid sprayStatusCharUuid = Guid('d61a1a55-0c02-4b6f-9c7e-2f6b6c1a9e10');

class BleException implements Exception {
  final String message;
  BleException(this.message);

  @override
  String toString() => message;
}

/// Shared app-wide (not per-screen) so the connection survives navigating
/// between screens, and any screen — notably the Home screen's emergency
/// stop — can see live connection/spray state without being the one that
/// made the connection.
class BleClient extends ChangeNotifier {
  BluetoothDevice? _device;
  BluetoothCharacteristic? _controlChar;
  StreamSubscription<BluetoothConnectionState>? _connectionSub;
  StreamSubscription<List<int>>? _statusSub;

  bool _spraying = false;
  String _status = '';

  BluetoothDevice? get device => _device;
  bool get isConnected => _device != null;
  bool get isSpraying => _spraying;
  String get status => _status;

  Future<Stream<List<ScanResult>>> startScan() async {
    try {
      await FlutterBluePlus.startScan(
        withServices: [sprayServiceUuid],
        timeout: const Duration(seconds: 15),
      );
    } on PlatformException catch (e) {
      if (e.message?.toLowerCase().contains('bluetooth must be turned on') == true) {
        throw BleException('Please turn on Bluetooth to scan for the drone.');
      }
      throw BleException('Could not start scan: ${e.message ?? e.code}');
    } catch (e) {
      throw BleException('Could not start scan: $e');
    }
    return FlutterBluePlus.scanResults;
  }

  Future<void> stopScan() => FlutterBluePlus.stopScan();

  Future<void> connect(BluetoothDevice device) async {
    await stopScan();
    try {
      // License.nonprofit: this is a personal/nonprofit drone project, per
      // flutter_blue_plus's own license categories for its connect() API.
      await device.connect(license: License.nonprofit, timeout: const Duration(seconds: 12));
    } catch (e) {
      throw BleException('Could not connect: $e');
    }

    final services = await device.discoverServices();
    BluetoothService? service;
    for (final s in services) {
      if (s.uuid == sprayServiceUuid) {
        service = s;
        break;
      }
    }
    if (service == null) {
      await device.disconnect();
      throw BleException('This device does not expose the spray control service.');
    }

    BluetoothCharacteristic? controlChar;
    BluetoothCharacteristic? statusChar;
    for (final c in service.characteristics) {
      if (c.uuid == sprayControlCharUuid) controlChar = c;
      if (c.uuid == sprayStatusCharUuid) statusChar = c;
    }
    if (controlChar == null || statusChar == null) {
      await device.disconnect();
      throw BleException('Spray control service is missing expected characteristics.');
    }

    _device = device;
    _controlChar = controlChar;
    _spraying = false;

    _connectionSub?.cancel();
    _connectionSub = device.connectionState.listen((state) {
      if (state == BluetoothConnectionState.disconnected) _reset();
    });

    try {
      _status = utf8.decode(await statusChar.read());
    } catch (_) {
      _status = '';
    }
    await statusChar.setNotifyValue(true);
    _statusSub?.cancel();
    _statusSub = statusChar.onValueReceived.listen((value) {
      _status = utf8.decode(value);
      notifyListeners();
    });

    notifyListeners();
  }

  Future<void> disconnect() async {
    await _device?.disconnect();
    _reset();
  }

  void _reset() {
    _connectionSub?.cancel();
    _statusSub?.cancel();
    _connectionSub = null;
    _statusSub = null;
    _device = null;
    _controlChar = null;
    _spraying = false;
    _status = '';
    notifyListeners();
  }

  Future<void> startSpraying() async {
    await _write('START');
    _spraying = true;
    notifyListeners();
  }

  Future<void> stopSpraying() async {
    await _write('STOP');
    _spraying = false;
    notifyListeners();
  }

  Future<void> _write(String command) async {
    final char = _controlChar;
    if (char == null) throw BleException('Not connected to a drone.');
    await char.write(utf8.encode(command), withoutResponse: false);
  }
}

/// Single instance shared across all screens — see class doc above.
final bleClient = BleClient();
