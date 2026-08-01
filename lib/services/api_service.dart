import 'dart:io';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';

Map<String, dynamic> _asMap(dynamic data, {Map<String, dynamic> fallback = const {}}) {
  if (data is Map<String, dynamic>) return data;
  if (data is Map) {
    return Map<String, dynamic>.from(data);
  }
  return fallback;
}

String _errMsg(DioException e) {
  try {
    final m = e.message;
    if (m == null || m.isEmpty) return e.toString();
    return m;
  } catch (_) {
    return 'Network error';
  }
}

class ApiService {
  // ── Single-finger backend (port 5002) ─────────────────────────────────────
  static const String _defaultSingleUrl = 'http://35.255.138.39:5002';
  static const String _singlePrefKey    = 'server_url_single_v1';

  // ── Slap (multi-finger) backend (port 5010) ───────────────────────────────
  static const String _defaultSlapUrl   = 'http://35.255.138.39:5010';
  static const String _slapPrefKey      = 'server_url_slap_v1';

  static String singleUrl = _defaultSingleUrl;
  static String slapUrl   = _defaultSlapUrl;

  static String get baseUrl => singleUrl;

  static final Dio _single = Dio(BaseOptions(
    baseUrl: _defaultSingleUrl,
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 90),
    responseType: ResponseType.json,
    validateStatus: (_) => true,
  ));

  static final Dio _slap = Dio(BaseOptions(
    baseUrl: _defaultSlapUrl,
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 90),
    responseType: ResponseType.json,
    validateStatus: (_) => true,
  ));

  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final savedSingle = prefs.getString(_singlePrefKey);
    if (savedSingle != null && savedSingle.isNotEmpty) {
      singleUrl = savedSingle;
      _single.options.baseUrl = savedSingle;
    }
    final savedSlap = prefs.getString(_slapPrefKey);
    if (savedSlap != null && savedSlap.isNotEmpty) {
      slapUrl = savedSlap;
      _slap.options.baseUrl = savedSlap;
    }
  }

  static Future<void> setSingleUrl(String url) async {
    singleUrl = url;
    _single.options.baseUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_singlePrefKey, url);
  }

  static Future<void> setSlapUrl(String url) async {
    slapUrl = url;
    _slap.options.baseUrl = url;
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_slapPrefKey, url);
  }

  static Future<void> setBaseUrl(String url) => setSingleUrl(url);

  // ── Health ────────────────────────────────────────────────────────────────
  static Future<Map<String, dynamic>> healthCheck({bool slap = false}) async {
    final r = await (slap ? _slap : _single).get('/health');
    return _asMap(r.data, fallback: <String, dynamic>{});
  }

  // ── Single-finger endpoints ───────────────────────────────────────────────
  static Future<Map<String, dynamic>> qualityCheck(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/quality_check', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 6)));
      return _asMap(r.data);
    } on DioException { return <String, dynamic>{}; }
  }

  static Future<Map<String, dynamic>> enroll({
    required String name, required String uid,
    required String batch, required File image,
  }) async {
    final fd = FormData.fromMap({
      'name': name, 'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/enroll', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> authenticate({
    required String batch, required File image,
  }) async {
    final fd = FormData.fromMap({
      'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/authenticate', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> verify({
    required String uid, required String batch, required File image,
  }) async {
    final fd = FormData.fromMap({
      'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/verify', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> process(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/process', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> readiness(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _single.post('/readiness', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> livenessGesture({
    required File image, required int expectedCount,
  }) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
      'expected_count': expectedCount.toString(),
    });
    try {
      final r = await _single.post('/liveness_gesture', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'success': false, 'error': 'Invalid response'});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<List<dynamic>> getUsers({String batch = ''}) async {
    final r = await _single.get('/users', queryParameters: {'batch': batch});
    final m = _asMap(r.data);
    final u = m['users'];
    if (u is List) return u;
    return <dynamic>[];
  }

  static Future<List<dynamic>> getHistory({String batch = ''}) async {
    final r = await _single.get('/history', queryParameters: {'batch': batch});
    final m = _asMap(r.data);
    final h = m['history'];
    if (h is List) return h;
    return <dynamic>[];
  }

  // ── Slap (multi-finger) endpoints ─────────────────────────────────────────
  static Future<Map<String, dynamic>> slapQualityCheck(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'slap.jpg'),
    });
    try {
      final r = await _slap.post('/quality_check', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 6)));
      return _asMap(r.data);
    } on DioException { return <String, dynamic>{}; }
  }

  static Future<Map<String, dynamic>> processSlap({
    required File image, String handSide = 'right', bool vis = true,
  }) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'slap.jpg'),
      'hand_side': handSide,
      'vis': vis ? '1' : '0',
    });
    try {
      final r = await _slap.post('/process_slap', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'error': 'Invalid response', 'offline': true});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'error': _errMsg(e), 'offline': true};
    }
  }

  static Future<Map<String, dynamic>> enrollSlap({
    required File image, required String name,
    required String uid, required String batch, String handSide = 'right',
  }) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'slap.jpg'),
      'name': name, 'uid': uid, 'batch': batch, 'hand_side': handSide,
    });
    try {
      final r = await _slap.post('/enroll_slap', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{'error': 'Invalid response', 'offline': true});
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'error': _errMsg(e), 'offline': true};
    }
  }

  static Future<Map<String, dynamic>> authenticateSlap({
    required String batch, required File image, String handSide = 'right',
  }) async {
    final fd = FormData.fromMap({
      'batch': batch, 'hand_side': handSide,
      'image': await MultipartFile.fromFile(image.path, filename: 'slap.jpg'),
    });
    try {
      final r = await _slap.post('/authenticate_slap', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{
            'success': false, 'error': 'Invalid response from server'
          });
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<Map<String, dynamic>> verifySlap({
    required String uid, required String batch,
    required File image, String handSide = 'right',
  }) async {
    final fd = FormData.fromMap({
      'uid': uid, 'batch': batch, 'hand_side': handSide,
      'image': await MultipartFile.fromFile(image.path, filename: 'slap.jpg'),
    });
    try {
      final r = await _slap.post('/verify_slap', data: fd);
      return _asMap(r.data,
          fallback: <String, dynamic>{
            'success': false, 'error': 'Invalid response from server'
          });
    } on DioException catch (e) {
      final d = _asMap(e.response?.data, fallback: <String, dynamic>{});
      if (d.isNotEmpty) return d;
      return <String, dynamic>{'success': false, 'error': _errMsg(e)};
    }
  }

  static Future<List<dynamic>> getSlapHistory({String batch = ''}) async {
    try {
      final r = await _slap.get('/history', queryParameters: {'batch': batch});
      final m = _asMap(r.data);
      final h = m['history'];
      if (h is List) return h;
      return <dynamic>[];
    } on DioException { return <dynamic>[]; }
  }
}
