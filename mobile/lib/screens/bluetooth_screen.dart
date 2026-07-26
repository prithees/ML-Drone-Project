import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';

import '../ble_client.dart';
import '../theme.dart';

class BluetoothScreen extends StatefulWidget {
  const BluetoothScreen({super.key});

  @override
  State<BluetoothScreen> createState() => _BluetoothScreenState();
}

class _BluetoothScreenState extends State<BluetoothScreen> {
  List<ScanResult> _results = [];
  bool _scanning = false;
  bool _connecting = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    if (!bleClient.isConnected) _startScan();
  }

  Future<void> _startScan() async {
    setState(() {
      _scanning = true;
      _error = null;
      _results = [];
    });
    try {
      final stream = await bleClient.startScan();
      if (!mounted) return;
      stream.listen(
        (results) {
          if (mounted) setState(() => _results = results);
        },
        onError: (e) {
          if (mounted) {
            setState(() {
              _error = 'Scan failed: $e';
              _scanning = false;
            });
          }
        },
      );
      FlutterBluePlus.isScanning.listen((scanning) {
        if (mounted) setState(() => _scanning = scanning);
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _scanning = false;
        });
      }
    }
  }

  Future<void> _connect(ScanResult result) async {
    setState(() {
      _connecting = true;
      _error = null;
    });
    try {
      await bleClient.connect(result.device);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _connecting = false);
    }
  }

  Future<void> _disconnect() async {
    await bleClient.disconnect();
    if (mounted) _startScan();
  }

  Future<void> _toggleSpraying() async {
    setState(() => _error = null);
    try {
      if (bleClient.isSpraying) {
        await bleClient.stopSpraying();
      } else {
        await bleClient.startSpraying();
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  String _labelFor(ScanResult result) {
    if (result.device.platformName.isNotEmpty) return result.device.platformName;
    if (result.advertisementData.advName.isNotEmpty) return result.advertisementData.advName;
    return result.device.remoteId.str;
  }

  @override
  void dispose() {
    // Deliberately does NOT disconnect bleClient — the connection is shared
    // app-wide so it survives navigating away from this screen.
    bleClient.stopScan();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: bleClient,
      builder: (context, _) {
        return Scaffold(
          appBar: AppBar(
            title: const Text('Drone Connection'),
            actions: [
              if (!bleClient.isConnected)
                IconButton(
                  icon: _scanning
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.refresh),
                  tooltip: 'Rescan',
                  onPressed: _scanning ? null : _startScan,
                ),
            ],
          ),
          body: bleClient.isConnected ? _connectedView() : _scanListView(),
        );
      },
    );
  }

  Widget _scanListView() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Text(
          'Scanning for the drone\'s Bluetooth control service. Tap a device to connect.',
          style: TextStyle(color: muted, fontSize: 13),
        ),
        const SizedBox(height: 16),
        if (_error != null) _banner(_error!, danger),
        if (_error != null) const SizedBox(height: 12),
        if (_scanning && _results.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 40),
            child: Center(child: CircularProgressIndicator()),
          )
        else if (_results.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 40),
            child: Center(
              child: Text('No devices found yet.', style: TextStyle(color: muted)),
            ),
          ),
        for (final result in _results)
          Card(
            child: ListTile(
              enabled: !_connecting,
              leading: const Icon(Icons.bluetooth, color: accent),
              title: Text(_labelFor(result)),
              subtitle: Text('${result.device.remoteId.str} · ${result.rssi} dBm'),
              trailing: _connecting
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.chevron_right),
              onTap: _connecting ? null : () => _connect(result),
            ),
          ),
      ],
    );
  }

  Widget _connectedView() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.bluetooth_connected, color: accent),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          bleClient.device?.platformName.isNotEmpty == true
                              ? bleClient.device!.platformName
                              : bleClient.device?.remoteId.str ?? '',
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 14),
                  Text('Status: ${bleClient.status}', style: const TextStyle(color: muted)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          if (_error != null) ...[_banner(_error!, danger), const SizedBox(height: 12)],
          ElevatedButton(
            onPressed: _toggleSpraying,
            style: bleClient.isSpraying
                ? ElevatedButton.styleFrom(backgroundColor: danger, foregroundColor: Colors.white)
                : null,
            child: Text(bleClient.isSpraying ? 'Stop Spraying' : 'Start Spraying'),
          ),
          const SizedBox(height: 12),
          OutlinedButton(
            onPressed: _disconnect,
            style: OutlinedButton.styleFrom(foregroundColor: muted, side: const BorderSide(color: border)),
            child: const Text('Disconnect'),
          ),
        ],
      ),
    );
  }

  Widget _banner(String message, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Text(message, style: TextStyle(color: color, fontSize: 13)),
    );
  }
}
