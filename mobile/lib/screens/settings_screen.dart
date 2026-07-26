import 'package:flutter/material.dart';

import '../api_client.dart';
import '../theme.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _controller = TextEditingController();
  bool _loading = true;
  bool _testing = false;
  String? _testResult;
  bool _testOk = false;

  double _threshold = 0.75;
  bool _thresholdLoading = true;
  bool _thresholdSaving = false;
  String? _thresholdError;
  bool _thresholdSaved = false;

  @override
  void initState() {
    super.initState();
    ApiClient.getBaseUrl().then((url) {
      if (!mounted) return;
      setState(() {
        _controller.text = url;
        _loading = false;
      });
    });
    _loadThreshold();
  }

  Future<void> _loadThreshold() async {
    try {
      final threshold = await ApiClient().getThreshold();
      if (mounted) setState(() => _threshold = threshold);
    } catch (e) {
      if (mounted) setState(() => _thresholdError = e.toString());
    } finally {
      if (mounted) setState(() => _thresholdLoading = false);
    }
  }

  Future<void> _saveThreshold() async {
    setState(() {
      _thresholdSaving = true;
      _thresholdError = null;
      _thresholdSaved = false;
    });
    try {
      final saved = await ApiClient().setThreshold(_threshold);
      if (mounted) {
        setState(() {
          _threshold = saved;
          _thresholdSaved = true;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _thresholdError = e.toString());
    } finally {
      if (mounted) setState(() => _thresholdSaving = false);
    }
  }

  Future<void> _save() async {
    await ApiClient.setBaseUrl(_controller.text);
    if (mounted) Navigator.of(context).pop();
  }

  Future<void> _testConnection() async {
    await ApiClient.setBaseUrl(_controller.text);
    if (!mounted) return;
    setState(() {
      _testing = true;
      _testResult = null;
    });
    try {
      final status = await ApiClient().getStatus();
      if (mounted) {
        setState(() {
          _testOk = true;
          _testResult = status.targetSet
              ? 'Connected. Target set with ${status.numReferences} reference photos.'
              : 'Connected. No target set yet.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _testOk = false;
          _testResult = e.toString();
        });
      }
    } finally {
      if (mounted) setState(() => _testing = false);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Server Settings')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(20),
              children: [
                const Text(
                  'Address of the Flask server running app.py. Use your computer\'s LAN IP '
                  '(e.g. http://192.168.1.20:5000) when testing on a physical device — '
                  'localhost only works from an emulator/simulator on the same machine.',
                  style: TextStyle(color: muted),
                ),
                const SizedBox(height: 16),
                TextField(
                  controller: _controller,
                  keyboardType: TextInputType.url,
                  decoration: const InputDecoration(
                    labelText: 'Server URL',
                    hintText: 'http://192.168.1.20:5000',
                  ),
                ),
                const SizedBox(height: 16),
                OutlinedButton(
                  onPressed: _testing ? null : _testConnection,
                  style: OutlinedButton.styleFrom(foregroundColor: accent, side: const BorderSide(color: border)),
                  child: _testing
                      ? const SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Test Connection'),
                ),
                if (_testResult != null) ...[
                  const SizedBox(height: 12),
                  Container(
                    padding: const EdgeInsets.all(14),
                    decoration: BoxDecoration(
                      color: (_testOk ? accent : danger).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: (_testOk ? accent : danger).withValues(alpha: 0.35)),
                    ),
                    child: Text(_testResult!, style: TextStyle(color: _testOk ? accent : danger)),
                  ),
                ],
                const SizedBox(height: 24),
                ElevatedButton(onPressed: _save, child: const Text('Save')),
                const Divider(height: 48, color: border),
                const Text(
                  'Match Threshold',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const SizedBox(height: 6),
                const Text(
                  'How similar a scan needs to be to the enrolled target before it counts '
                  'as a match. Lower catches more (with more false positives); higher is '
                  'stricter. Applies immediately to both the drone loop and manual scans.',
                  style: TextStyle(color: muted, fontSize: 13),
                ),
                const SizedBox(height: 12),
                if (_thresholdLoading)
                  const Center(child: CircularProgressIndicator())
                else ...[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('similarity cutoff', style: TextStyle(color: muted, fontSize: 12)),
                      Text(
                        '${(_threshold * 100).toStringAsFixed(0)}%',
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  Slider(
                    value: _threshold,
                    min: 0.5,
                    max: 1.0,
                    divisions: 50,
                    activeColor: accent,
                    label: '${(_threshold * 100).toStringAsFixed(0)}%',
                    onChanged: (value) => setState(() {
                      _threshold = value;
                      _thresholdSaved = false;
                    }),
                  ),
                  const SizedBox(height: 8),
                  if (_thresholdError != null) ...[
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: danger.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: danger.withValues(alpha: 0.35)),
                      ),
                      child: Text(_thresholdError!, style: const TextStyle(color: danger, fontSize: 13)),
                    ),
                    const SizedBox(height: 12),
                  ],
                  if (_thresholdSaved) ...[
                    const Text('Threshold saved.', style: TextStyle(color: accent, fontSize: 13)),
                    const SizedBox(height: 8),
                  ],
                  OutlinedButton(
                    onPressed: _thresholdSaving ? null : _saveThreshold,
                    style: OutlinedButton.styleFrom(foregroundColor: accent, side: const BorderSide(color: border)),
                    child: _thresholdSaving
                        ? const SizedBox(
                            height: 18,
                            width: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Text('Save Threshold'),
                  ),
                ],
              ],
            ),
    );
  }
}
