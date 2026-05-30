import 'dart:io';
import 'package:dio/dio.dart';

class ApiService {
  static String baseUrl = 'http://192.168.101.3:5001';

  static final Dio _dio = Dio(BaseOptions(
    baseUrl: baseUrl,
    connectTimeout: const Duration(seconds: 30),
    receiveTimeout: const Duration(seconds: 90),
  ));

  static void setBaseUrl(String url) {
    baseUrl = url;
    _dio.options.baseUrl = url;
  }

  static Future<Map<String, dynamic>> healthCheck() async {
    final r = await _dio.get('/health');
    return r.data;
  }

  static Future<Map<String, dynamic>> qualityCheck(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/quality_check', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 5)));
      return r.data;
    } on DioException {
      return {};
    }
  }

  static Future<Map<String, dynamic>> enroll({
    required String name,
    required String uid,
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'name': name, 'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/enroll', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> authenticate({
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/authenticate', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> verify({
    required String uid,
    required String batch,
    required File image,
  }) async {
    final fd = FormData.fromMap({
      'uid': uid, 'batch': batch,
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/verify', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> process(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/process', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<List<dynamic>> getUsers({String batch = ''}) async {
    final r = await _dio.get('/users', queryParameters: {'batch': batch});
    return r.data['users'] ?? [];
  }

  static Future<List<dynamic>> getHistory({String batch = ''}) async {
    final r = await _dio.get('/history', queryParameters: {'batch': batch});
    return r.data['history'] ?? [];
  }

  static Future<Map<String, dynamic>> readiness(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/readiness', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }

  static Future<Map<String, dynamic>> checkRoi(File image) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
    });
    try {
      final r = await _dio.post('/check_roi', data: fd,
          options: Options(receiveTimeout: const Duration(seconds: 5)));
      return r.data;
    } on DioException {
      return {};
    }
  }

  static Future<Map<String, dynamic>> livenessGesture({
    required File image,
    required int expectedCount,
  }) async {
    final fd = FormData.fromMap({
      'image': await MultipartFile.fromFile(image.path, filename: 'fp.jpg'),
      'expected_count': expectedCount.toString(),
    });
    try {
      final r = await _dio.post('/liveness_gesture', data: fd);
      return r.data;
    } on DioException catch (e) {
      return e.response?.data ?? {'success': false, 'error': e.message};
    }
  }
}
