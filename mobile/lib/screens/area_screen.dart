import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:geolocator/geolocator.dart';
import 'package:latlong2/latlong.dart';

import '../api_client.dart';
import '../theme.dart';

const _defaultCenter = LatLng(20.5937, 78.9629);
const _defaultZoom = 4.0;
const _locatedZoom = 18.0;

class AreaScreen extends StatefulWidget {
  const AreaScreen({super.key});

  @override
  State<AreaScreen> createState() => _AreaScreenState();
}

class _AreaScreenState extends State<AreaScreen> {
  final _api = ApiClient();
  final _mapController = MapController();

  List<LatLng> _points = [];
  bool _loading = true;
  bool _saving = false;
  bool _locating = false;
  String? _error;
  String? _info;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final area = await _api.getArea();
      if (area.areaSet && mounted) {
        setState(() => _points = area.points);
      }
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
    if (mounted) _centerOnMyLocation(showErrors: false);
  }

  Future<void> _centerOnMyLocation({bool showErrors = true}) async {
    setState(() {
      _locating = true;
      if (showErrors) _error = null;
    });
    try {
      if (!await Geolocator.isLocationServiceEnabled()) {
        if (showErrors) _error = 'Location services are turned off.';
        return;
      }
      var permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied || permission == LocationPermission.deniedForever) {
        if (showErrors) _error = 'Location permission denied.';
        return;
      }
      final position = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(accuracy: LocationAccuracy.high),
      );
      if (mounted) _mapController.move(LatLng(position.latitude, position.longitude), _locatedZoom);
    } catch (e) {
      if (showErrors && mounted) setState(() => _error = 'Could not get current location: $e');
    } finally {
      if (mounted) setState(() => _locating = false);
    }
  }

  void _onMapTap(TapPosition tapPosition, LatLng point) {
    setState(() {
      _points = [..._points, point];
      _error = null;
      _info = null;
    });
  }

  void _undoLastPoint() {
    if (_points.isEmpty) return;
    setState(() => _points = _points.sublist(0, _points.length - 1));
  }

  void _clearPoints() {
    setState(() {
      _points = [];
      _info = null;
      _error = null;
    });
  }

  Future<void> _save() async {
    if (_points.length < 3) {
      setState(() => _error = 'Tap at least 3 points on the map to form a boundary.');
      return;
    }
    setState(() {
      _saving = true;
      _error = null;
      _info = null;
    });
    try {
      await _api.setArea(_points);
      if (mounted) setState(() => _info = 'Spray area saved (${_points.length} points).');
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Spray Area'),
        actions: [
          IconButton(
            icon: _locating
                ? const SizedBox(
                    height: 18,
                    width: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.my_location),
            tooltip: 'Center on my location',
            onPressed: _locating ? null : () => _centerOnMyLocation(),
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
                  child: Text(
                    'Tap the map to place boundary points (at least 3), in order '
                    'around the edge of the area the drone is allowed to spray.',
                    style: const TextStyle(color: muted, fontSize: 13),
                  ),
                ),
                Expanded(
                  child: Stack(
                    children: [
                      FlutterMap(
                        mapController: _mapController,
                        options: MapOptions(
                          initialCenter: _points.isNotEmpty ? _points.first : _defaultCenter,
                          initialZoom: _points.isNotEmpty ? _locatedZoom : _defaultZoom,
                          onTap: _onMapTap,
                        ),
                        children: [
                          TileLayer(
                            urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                            userAgentPackageName: 'com.leafscan.leaf_scan_app',
                          ),
                          if (_points.length >= 3)
                            PolygonLayer(
                              polygons: [
                                Polygon(
                                  points: _points,
                                  color: accent.withValues(alpha: 0.25),
                                  borderColor: accent,
                                  borderStrokeWidth: 2,
                                ),
                              ],
                            ),
                          if (_points.length == 2)
                            PolylineLayer(
                              polylines: [
                                Polyline(points: _points, color: accent, strokeWidth: 2),
                              ],
                            ),
                          MarkerLayer(
                            markers: [
                              for (var i = 0; i < _points.length; i++)
                                Marker(
                                  point: _points[i],
                                  width: 28,
                                  height: 28,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      color: accent,
                                      shape: BoxShape.circle,
                                      border: Border.all(color: const Color(0xFF06231A), width: 2),
                                    ),
                                    alignment: Alignment.center,
                                    child: Text(
                                      '${i + 1}',
                                      style: const TextStyle(
                                        color: Color(0xFF06231A),
                                        fontSize: 11,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                ),
                            ],
                          ),
                        ],
                      ),
                      Positioned(
                        left: 12,
                        top: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                          decoration: BoxDecoration(
                            color: bg0.withValues(alpha: 0.85),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: border),
                          ),
                          child: Text(
                            '${_points.length} point${_points.length == 1 ? '' : 's'}',
                            style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      if (_error != null) _banner(_error!, danger),
                      if (_info != null) _banner(_info!, accent),
                      if (_error != null || _info != null) const SizedBox(height: 10),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton(
                              onPressed: _points.isEmpty ? null : _undoLastPoint,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: accent,
                                side: const BorderSide(color: border),
                              ),
                              child: const Text('Undo point'),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: OutlinedButton(
                              onPressed: _points.isEmpty ? null : _clearPoints,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: danger,
                                side: const BorderSide(color: border),
                              ),
                              child: const Text('Clear'),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      ElevatedButton(
                        onPressed: _saving ? null : _save,
                        child: _saving
                            ? const SizedBox(
                                height: 18,
                                width: 18,
                                child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF06231A)),
                              )
                            : const Text('Save Spray Area'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  Widget _banner(String message, Color color) {
    return Container(
      margin: const EdgeInsets.only(bottom: 4),
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
