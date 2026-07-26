import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../api_client.dart';
import '../theme.dart';

const _defaultCenter = LatLng(20.5937, 78.9629);
const _defaultZoom = 4.0;
const _locatedZoom = 16.0;

class HistoryMapScreen extends StatefulWidget {
  const HistoryMapScreen({super.key});

  @override
  State<HistoryMapScreen> createState() => _HistoryMapScreenState();
}

class _HistoryMapScreenState extends State<HistoryMapScreen> {
  final _api = ApiClient();
  bool _loading = true;
  String? _error;
  List<HistoryEntry> _geotagged = [];
  List<LatLng> _areaPoints = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final results = await Future.wait([_api.getHistory(limit: 200), _api.getArea()]);
      final entries = results[0] as List<HistoryEntry>;
      final area = results[1] as SprayArea;
      if (mounted) {
        setState(() {
          _geotagged = entries.where((e) => e.hasLocation).toList();
          _areaPoints = area.points;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Color _colorFor(HistoryEntry entry) {
    if (entry.sprayed) return warn;
    if (entry.matched) return accent;
    return muted;
  }

  @override
  Widget build(BuildContext context) {
    final center = _geotagged.isNotEmpty
        ? LatLng(_geotagged.first.lat!, _geotagged.first.lng!)
        : (_areaPoints.isNotEmpty ? _areaPoints.first : _defaultCenter);
    final zoom = (_geotagged.isNotEmpty || _areaPoints.isNotEmpty) ? _locatedZoom : _defaultZoom;

    return Scaffold(
      appBar: AppBar(title: const Text('History Map')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                if (_error != null)
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: Text(_error!, style: const TextStyle(color: danger)),
                  ),
                if (_geotagged.isEmpty && _error == null)
                  const Padding(
                    padding: EdgeInsets.all(16),
                    child: Text(
                      'No geotagged scans yet — only scans made by the drone with a GPS fix show up here.',
                      style: TextStyle(color: muted, fontSize: 13),
                    ),
                  ),
                Expanded(
                  child: FlutterMap(
                    options: MapOptions(initialCenter: center, initialZoom: zoom),
                    children: [
                      TileLayer(
                        urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        userAgentPackageName: 'com.leafscan.leaf_scan_app',
                      ),
                      if (_areaPoints.length >= 3)
                        PolygonLayer(
                          polygons: [
                            Polygon(
                              points: _areaPoints,
                              color: accent.withValues(alpha: 0.12),
                              borderColor: accent,
                              borderStrokeWidth: 2,
                            ),
                          ],
                        ),
                      MarkerLayer(
                        markers: [
                          for (final entry in _geotagged)
                            Marker(
                              point: LatLng(entry.lat!, entry.lng!),
                              width: 20,
                              height: 20,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: _colorFor(entry),
                                  shape: BoxShape.circle,
                                  border: Border.all(color: const Color(0xFF06231A), width: 2),
                                ),
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: Wrap(
                    spacing: 16,
                    children: [
                      _legendDot(warn, 'Sprayed'),
                      _legendDot(accent, 'Matched'),
                      _legendDot(muted, 'No match'),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(fontSize: 12, color: muted)),
      ],
    );
  }
}
