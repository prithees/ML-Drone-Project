import 'package:flutter/material.dart';

import '../api_client.dart';
import '../ble_client.dart';
import '../theme.dart';
import 'area_screen.dart';
import 'bluetooth_screen.dart';
import 'enroll_screen.dart';
import 'history_screen.dart';
import 'remote_control_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _api = ApiClient();
  late Future<TargetStatus> _statusFuture;

  @override
  void initState() {
    super.initState();
    _statusFuture = _api.getStatus();
  }

  void _refresh() {
    setState(() {
      _statusFuture = _api.getStatus();
    });
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: bleClient,
      builder: (context, _) => _buildScaffold(context),
    );
  }

  Widget _buildScaffold(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.asset('assets/icon/icon.png', width: 28, height: 28),
            ),
            const SizedBox(width: 10),
            const Text('AgroScan'),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Server settings',
            onPressed: () async {
              await Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const SettingsScreen()),
              );
              _refresh();
            },
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: () async => _refresh(),
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            if (bleClient.isConnected) ...[_emergencyStopBar(context), const SizedBox(height: 16)],
            const Text(
              'Smart Drone Pesticide Spraying',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 6),
            const Text(
              'No fixed plant list, no training. Show it a few example photos of what to spray, then scan new photos against that target.',
              style: TextStyle(color: muted),
            ),
            const SizedBox(height: 20),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(18),
                child: FutureBuilder<TargetStatus>(
                  future: _statusFuture,
                  builder: (context, snapshot) {
                    if (snapshot.connectionState != ConnectionState.done) {
                      return const Padding(
                        padding: EdgeInsets.symmetric(vertical: 12),
                        child: Center(child: CircularProgressIndicator()),
                      );
                    }
                    if (snapshot.hasError) {
                      return _statusError(snapshot.error.toString());
                    }
                    final status = snapshot.data!;
                    return _statusContent(status);
                  },
                ),
              ),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: () async {
                await Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const EnrollScreen()),
                );
                _refresh();
              },
              icon: const Icon(Icons.collections_outlined),
              label: const Text('Set Target'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                foregroundColor: accent,
                side: const BorderSide(color: border),
              ),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const AreaScreen()),
                );
              },
              icon: const Icon(Icons.map_outlined),
              label: const Text('Configure Spray Area'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                foregroundColor: accent,
                side: const BorderSide(color: border),
              ),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const BluetoothScreen()),
                );
              },
              icon: const Icon(Icons.bluetooth),
              label: const Text('Connect to Drone'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                foregroundColor: accent,
                side: const BorderSide(color: border),
              ),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const RemoteControlScreen()),
                );
              },
              icon: const Icon(Icons.settings_remote),
              label: const Text('Remote Control (Server)'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              style: OutlinedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 14),
                foregroundColor: accent,
                side: const BorderSide(color: border),
              ),
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const HistoryScreen()),
                );
              },
              icon: const Icon(Icons.history),
              label: const Text('Scan History'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _emergencyStopBar(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: () async {
          final messenger = ScaffoldMessenger.of(context);
          try {
            await bleClient.stopSpraying();
            messenger.showSnackBar(const SnackBar(content: Text('Stop sent to drone.')));
          } catch (e) {
            messenger.showSnackBar(SnackBar(content: Text('Could not send stop: $e')));
          }
        },
        style: ElevatedButton.styleFrom(
          backgroundColor: danger,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
        ),
        icon: const Icon(Icons.stop_circle_outlined),
        label: const Text('EMERGENCY STOP', style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5)),
      ),
    );
  }

  Widget _statusError(String message) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Could not load target status', style: TextStyle(fontWeight: FontWeight.bold, color: danger)),
        const SizedBox(height: 6),
        Text(message, style: const TextStyle(color: muted, fontSize: 13)),
      ],
    );
  }

  Widget _statusContent(TargetStatus status) {
    if (!status.targetSet) {
      return const Text(
        'No target set yet. Tap "Set Target" below to define what the drone should look for.',
        style: TextStyle(color: muted),
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
          decoration: BoxDecoration(
            color: accent.withValues(alpha: 0.12),
            borderRadius: BorderRadius.circular(999),
          ),
          child: const Text('Target set', style: TextStyle(color: accent, fontWeight: FontWeight.bold)),
        ),
        const SizedBox(height: 12),
        Text(
          'Learned from ${status.numReferences} reference photo${status.numReferences != 1 ? 's' : ''}.',
          style: const TextStyle(color: muted),
        ),
        const SizedBox(height: 10),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: const Color(0xFF0F1B17),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: const Color(0xFF17281F)),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Action when matched', style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 6),
              Text(status.guidance ?? '', style: const TextStyle(color: muted)),
            ],
          ),
        ),
      ],
    );
  }
}
