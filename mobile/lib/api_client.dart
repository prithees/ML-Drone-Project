import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:latlong2/latlong.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Default base URL per platform. Android emulators route 10.0.2.2 to the
/// host machine's localhost; everything else (iOS simulator, physical
/// devices on the same network) should point at the host's LAN IP, which
/// the user can change from the settings screen.
String defaultBaseUrl() {
  if (Platform.isAndroid) return 'http://10.0.2.2:5000';
  return 'http://localhost:5000';
}

class ApiException implements Exception {
  final String message;
  ApiException(this.message);

  @override
  String toString() => message;
}

class TargetStatus {
  final bool targetSet;
  final int numReferences;
  final String? guidance;

  TargetStatus({required this.targetSet, required this.numReferences, this.guidance});

  factory TargetStatus.fromJson(Map<String, dynamic> json) {
    return TargetStatus(
      targetSet: json['target_set'] as bool,
      numReferences: json['num_references'] as int,
      guidance: json['guidance'] as String?,
    );
  }
}

class EnrollResult {
  final int numReferences;
  final String guidance;

  EnrollResult({required this.numReferences, required this.guidance});

  factory EnrollResult.fromJson(Map<String, dynamic> json) {
    return EnrollResult(
      numReferences: json['num_references'] as int,
      guidance: json['guidance'] as String,
    );
  }
}

class SprayArea {
  final bool areaSet;
  final List<LatLng> points;

  SprayArea({required this.areaSet, required this.points});

  factory SprayArea.fromJson(Map<String, dynamic> json) {
    final rawPoints = json['points'] as List<dynamic>;
    return SprayArea(
      areaSet: json['area_set'] as bool,
      points: rawPoints
          .map((p) => LatLng((p['lat'] as num).toDouble(), (p['lng'] as num).toDouble()))
          .toList(),
    );
  }
}

class HistoryEntry {
  final DateTime timestamp;
  final String source;
  final bool matched;
  final double score;
  final bool sprayed;
  final double? lat;
  final double? lng;

  HistoryEntry({
    required this.timestamp,
    required this.source,
    required this.matched,
    required this.score,
    required this.sprayed,
    this.lat,
    this.lng,
  });

  bool get hasLocation => lat != null && lng != null;

  factory HistoryEntry.fromJson(Map<String, dynamic> json) {
    return HistoryEntry(
      timestamp: DateTime.fromMillisecondsSinceEpoch(((json['timestamp'] as num) * 1000).round()),
      source: json['source'] as String,
      matched: json['matched'] as bool,
      score: (json['score'] as num).toDouble(),
      sprayed: json['sprayed'] as bool,
      lat: (json['lat'] as num?)?.toDouble(),
      lng: (json['lng'] as num?)?.toDouble(),
    );
  }
}

class ApiClient {
  static const _prefsKey = 'base_url';

  static Future<String> getBaseUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_prefsKey) ?? defaultBaseUrl();
  }

  static Future<void> setBaseUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    // Strip a trailing slash so endpoint paths don't end up with "//".
    final trimmed = url.trim().replaceFirst(RegExp(r'/+$'), '');
    await prefs.setString(_prefsKey, trimmed);
  }

  Future<TargetStatus> getStatus() async {
    final baseUrl = await getBaseUrl();
    final response = await _get(Uri.parse('$baseUrl/api/status'));
    return TargetStatus.fromJson(response);
  }

  Future<EnrollResult> enroll({required List<XFile> files, required String guidance}) async {
    final baseUrl = await getBaseUrl();
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/api/enroll'));
    request.fields['guidance'] = guidance;
    for (final file in files) {
      request.files.add(await http.MultipartFile.fromPath('file', file.path));
    }
    final response = await _send(request);
    return EnrollResult.fromJson(response);
  }

  Future<SprayArea> getArea() async {
    final baseUrl = await getBaseUrl();
    final response = await _get(Uri.parse('$baseUrl/api/area'));
    return SprayArea.fromJson(response);
  }

  Future<SprayArea> setArea(List<LatLng> points) async {
    final baseUrl = await getBaseUrl();
    await _postJson(Uri.parse('$baseUrl/api/area'), {
      'points': points.map((p) => {'lat': p.latitude, 'lng': p.longitude}).toList(),
    });
    return SprayArea(areaSet: true, points: points);
  }

  Future<List<HistoryEntry>> getHistory({int? limit}) async {
    final baseUrl = await getBaseUrl();
    final uri = Uri.parse('$baseUrl/api/history').replace(
      queryParameters: limit != null ? {'limit': '$limit'} : null,
    );
    final response = await _get(uri);
    final entries = response['entries'] as List<dynamic>;
    return entries.map((e) => HistoryEntry.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<double> getThreshold() async {
    final baseUrl = await getBaseUrl();
    final response = await _get(Uri.parse('$baseUrl/api/threshold'));
    return (response['threshold'] as num).toDouble();
  }

  Future<double> setThreshold(double threshold) async {
    final baseUrl = await getBaseUrl();
    final response = await _postJson(Uri.parse('$baseUrl/api/threshold'), {'threshold': threshold});
    return (response['threshold'] as num).toDouble();
  }

  Future<Map<String, dynamic>> _postJson(Uri uri, Map<String, dynamic> body) async {
    http.Response response;
    try {
      response = await http
          .post(uri, headers: {'Content-Type': 'application/json'}, body: jsonEncode(body))
          .timeout(const Duration(seconds: 20));
    } catch (_) {
      throw ApiException('Could not reach the server at $uri. Check the server address in Settings.');
    }
    return _decode(response);
  }

  /// Alternative to BleClient's start/stop for controlling the drone over
  /// HTTP instead of Bluetooth — works over any network the server is
  /// reachable on (WiFi, or a cellular-connected Pi), not just short-range
  /// BLE. Only one of the two should be driving the drone at a time; see
  /// the comment above the /api/spray/* routes in app.py.
  Future<String> startSpray() async {
    final baseUrl = await getBaseUrl();
    final response = await _postEmpty(Uri.parse('$baseUrl/api/spray/start'));
    return response['status'] as String;
  }

  Future<String> stopSpray() async {
    final baseUrl = await getBaseUrl();
    final response = await _postEmpty(Uri.parse('$baseUrl/api/spray/stop'));
    return response['status'] as String;
  }

  Future<String> getSprayStatus() async {
    final baseUrl = await getBaseUrl();
    final response = await _get(Uri.parse('$baseUrl/api/spray/status'));
    return response['status'] as String;
  }

  Future<Map<String, dynamic>> _postEmpty(Uri uri) async {
    http.Response response;
    try {
      response = await http.post(uri).timeout(const Duration(seconds: 20));
    } catch (_) {
      throw ApiException('Could not reach the server at $uri. Check the server address in Settings.');
    }
    return _decode(response);
  }

  Future<Map<String, dynamic>> _get(Uri uri) async {
    http.Response response;
    try {
      response = await http.get(uri).timeout(const Duration(seconds: 20));
    } catch (_) {
      throw ApiException('Could not reach the server at $uri. Check the server address in Settings.');
    }
    return _decode(response);
  }

  Future<Map<String, dynamic>> _send(http.MultipartRequest request) async {
    http.StreamedResponse streamed;
    try {
      streamed = await request.send().timeout(const Duration(seconds: 60));
    } catch (_) {
      throw ApiException('Could not reach the server at ${request.url}. Check the server address in Settings.');
    }
    final response = await http.Response.fromStream(streamed);
    return _decode(response);
  }

  Map<String, dynamic> _decode(http.Response response) {
    Map<String, dynamic> body;
    try {
      body = jsonDecode(response.body) as Map<String, dynamic>;
    } catch (_) {
      throw ApiException('Unexpected response from server (status ${response.statusCode}).');
    }
    if (response.statusCode >= 400) {
      throw ApiException(body['error'] as String? ?? 'Request failed (status ${response.statusCode}).');
    }
    return body;
  }
}
