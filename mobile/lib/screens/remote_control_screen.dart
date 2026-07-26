import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';

/// Controls the drone over the same HTTP API used for enroll/scan/area,
/// instead of Bluetooth — this is the control path that works beyond BLE's
/// short range (e.g. a cellular-connected Pi + a tunnel), as long as the
/// server address in Settings actually reaches it. See the comment above
/// the /api/spray/* routes in app.py: don't run this against a Pi that's
/// also running ble_server.py at the same time.
class RemoteControlScreen extends StatefulWidget {
  const RemoteControlScreen({super.key});

  @override
  State<RemoteControlScreen> createState() => _RemoteControlScreenState();
}

class _RemoteControlScreenState extends State<RemoteControlScreen> {
  final _api = ApiClient();
  String _status = '';
  bool _spraying = false;
  bool _busy = false;
  bool _loadingStatus = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _refreshStatus();
  }

  Future<void> _refreshStatus() async {
    setState(() {
      _loadingStatus = true;
      _error = null;
    });
    try {
      final status = await _api.getSprayStatus();
      if (mounted) {
        setState(() {
          _status = status;
          _spraying = status.startsWith('spraying') || status == 'scanning';
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loadingStatus = false);
    }
  }

  Future<void> _toggleSpraying() async {
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final status = _spraying ? await _api.stopSpray() : await _api.startSpray();
      if (mounted) {
        setState(() {
          _status = status;
          _spraying = !_spraying;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Remote Control'),
        actions: [
          IconButton(
            icon: _loadingStatus
                ? const SizedBox(height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Icon(Icons.refresh),
            tooltip: 'Refresh status',
            onPressed: _loadingStatus ? null : _refreshStatus,
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Controls the drone over the server address configured in Settings — '
              'works over any network the server is reachable on, not just Bluetooth range.',
              style: TextStyle(color: muted, fontSize: 13),
            ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(18),
                child: Text('Status: $_status', style: const TextStyle(color: muted)),
              ),
            ),
            const SizedBox(height: 16),
            if (_error != null) ...[
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: danger.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: danger.withValues(alpha: 0.35)),
                ),
                child: Text(_error!, style: const TextStyle(color: danger, fontSize: 13)),
              ),
              const SizedBox(height: 12),
            ],
            ElevatedButton(
              onPressed: _busy ? null : _toggleSpraying,
              style: _spraying
                  ? ElevatedButton.styleFrom(backgroundColor: danger, foregroundColor: Colors.white)
                  : null,
              child: _busy
                  ? const SizedBox(
                      height: 18,
                      width: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF06231A)),
                    )
                  : Text(_spraying ? 'Stop Spraying' : 'Start Spraying'),
            ),
          ],
        ),
      ),
    );
  }
}
