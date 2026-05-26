import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../services/api_service.dart';

class FingerprintCameraWidget extends StatefulWidget {
  final void Function(File) onImageCaptured;
  final bool disabled;
  const FingerprintCameraWidget({
    super.key,
    required this.onImageCaptured,
    this.disabled = false,
  });

  @override
  State<FingerprintCameraWidget> createState() => _FingerprintCameraWidgetState();
}

class _FingerprintCameraWidgetState extends State<FingerprintCameraWidget>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {

  // ── Camera state ────────────────────────────────────────────────────────────
  CameraController? _ctrl;
  bool _initializing = false;
  bool _live = false;
  bool _flashing = false;
  bool _torchOn = false;
  File? _captured;
  String? _error;
  double _minZoom = 1.0, _maxZoom = 1.0, _zoom = 1.0;
  Offset? _focusPt;
  Timer? _focusTimer;

  // ── Auto-capture state ──────────────────────────────────────────────────────
  bool _autoCapture = true;          // toggle — user can switch to manual
  Timer? _qualityTimer;
  bool _polling = false;             // prevent overlapping polls
  String _guidance = 'Place finger inside the oval';
  Color _guidanceColor = Colors.white60;
  double _qualityScore = 0.0;        // 0.0 – 1.0 for progress bar
  int _passCount = 0;                // consecutive passes needed = 3
  static const int _passesNeeded = 3;
  static const Duration _pollInterval = Duration(milliseconds: 700);

  // ── Scan animation ──────────────────────────────────────────────────────────
  late AnimationController _scanCtrl;
  late Animation<double> _scanAnim;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _scanCtrl = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);
    _scanAnim = Tween<double>(begin: 0, end: 1).animate(
        CurvedAnimation(parent: _scanCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _focusTimer?.cancel();
    _qualityTimer?.cancel();
    _scanCtrl.dispose();
    _ctrl?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) _stopCamera();
  }

  // ── Camera ──────────────────────────────────────────────────────────────────

  Future<void> _startCamera() async {
    if (widget.disabled || _initializing) return;
    setState(() { _initializing = true; _error = null; });
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) throw Exception('No camera found');
      final cam = cams.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cams.first);
      _ctrl = CameraController(cam, ResolutionPreset.max,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
      await _ctrl!.initialize();
      await _ctrl!.setFocusMode(FocusMode.auto);
      await _ctrl!.setExposureMode(ExposureMode.auto);
      _minZoom = await _ctrl!.getMinZoomLevel();
      _maxZoom = await _ctrl!.getMaxZoomLevel();
      _zoom = _minZoom;
      try { await _ctrl!.setFlashMode(FlashMode.off); } catch (_) {}
      if (mounted) {
        setState(() { _live = true; _initializing = false; });
        if (_autoCapture) _startQualityPolling();
      }
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _initializing = false; });
    }
  }

  Future<void> _stopCamera() async {
    _stopQualityPolling();
    if (_torchOn) { try { await _ctrl!.setFlashMode(FlashMode.off); } catch (_) {} }
    await _ctrl?.dispose();
    _ctrl = null;
    if (mounted) {
      setState(() {
        _live = false; _torchOn = false; _captured = null;
        _guidance = 'Place finger inside the oval';
        _guidanceColor = Colors.white60;
        _qualityScore = 0.0; _passCount = 0;
      });
    }
  }

  Future<void> _toggleTorch() async {
    if (_ctrl == null) return;
    try {
      await _ctrl!.setFlashMode(_torchOn ? FlashMode.off : FlashMode.torch);
      setState(() => _torchOn = !_torchOn);
    } catch (_) {}
  }

  Future<void> _onTap(TapDownDetails d, BoxConstraints c) async {
    if (_ctrl == null || !_live) return;
    final x = (d.localPosition.dx / c.maxWidth).clamp(0.0, 1.0);
    final y = (d.localPosition.dy / c.maxHeight).clamp(0.0, 1.0);
    setState(() => _focusPt = d.localPosition);
    _focusTimer?.cancel();
    _focusTimer = Timer(const Duration(milliseconds: 1500), () {
      if (mounted) setState(() => _focusPt = null);
    });
    try {
      await _ctrl!.setFocusPoint(Offset(x, y));
      await _ctrl!.setExposurePoint(Offset(x, y));
    } catch (_) {}
  }

  Future<void> _capture() async {
    if (_ctrl == null || !_ctrl!.value.isInitialized) return;
    _stopQualityPolling();
    await Future.delayed(const Duration(milliseconds: 300));
    setState(() => _flashing = true);
    await Future.delayed(const Duration(milliseconds: 120));
    setState(() => _flashing = false);
    try {
      final xf = await _ctrl!.takePicture();
      if (mounted) setState(() { _captured = File(xf.path); _live = false; });
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Capture failed: $e');
        if (_autoCapture) _startQualityPolling();
      }
    }
  }

  void _useImage() { if (_captured != null) widget.onImageCaptured(_captured!); }

  Future<void> _retry() async {
    setState(() {
      _captured = null; _live = false;
      _passCount = 0; _qualityScore = 0.0;
      _guidance = 'Place finger inside the oval';
      _guidanceColor = Colors.white60;
    });
    await _startCamera();
  }

  // ── Auto-capture quality polling ────────────────────────────────────────────

  void _startQualityPolling() {
    _qualityTimer?.cancel();
    _passCount = 0;
    _qualityTimer = Timer.periodic(_pollInterval, (_) => _pollQuality());
  }

  void _stopQualityPolling() {
    _qualityTimer?.cancel();
    _qualityTimer = null;
    _polling = false;
  }

  Future<void> _pollQuality() async {
    if (!_live || _ctrl == null || !_ctrl!.value.isInitialized) return;
    if (_polling || _captured != null) return;
    _polling = true;

    try {
      // Take a quick low-res snapshot for quality check
      final xf = await _ctrl!.takePicture();
      final file = File(xf.path);
      final result = await ApiService.qualityCheck(file);

      if (!mounted) return;

      final passed = result['passed'] == true;
      final guidance = result['guidance'] as String? ?? 'Checking...';
      final blur = (result['blur']?['blur_score'] as num?)?.toDouble() ?? 0;
      // Normalize blur score to 0-1 (fingerprints score 2-70, good = above 5)
      final score = (blur / 30.0).clamp(0.0, 1.0);

      setState(() {
        _guidance = passed ? '✓  Good — hold still' : guidance;
        _guidanceColor = passed ? YS.amber : Colors.orangeAccent;
        _qualityScore = score;
      });

      if (passed) {
        _passCount++;
        if (_passCount >= _passesNeeded) {
          _stopQualityPolling();
          await _capture();
        }
      } else {
        _passCount = 0;
      }
    } catch (_) {
      // Server unreachable — fall back to manual silently
      if (mounted) {
        setState(() {
          _guidance = 'Auto-check unavailable — capture manually';
          _guidanceColor = Colors.white38;
        });
      }
    } finally {
      _polling = false;
    }
  }

  void _toggleAutoCapture() {
    setState(() => _autoCapture = !_autoCapture);
    if (_live) {
      if (_autoCapture) {
        _startQualityPolling();
      } else {
        _stopQualityPolling();
        setState(() {
          _guidance = 'Place finger inside the oval';
          _guidanceColor = Colors.white60;
          _qualityScore = 0.0;
          _passCount = 0;
        });
      }
    }
  }

  // ── Build ───────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: YS.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: YS.stroke),
      ),
      clipBehavior: Clip.hardEdge,
      child: Column(children: [_header(), _viewfinder(), _controls()]),
    );
  }

  Widget _header() => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
    decoration: const BoxDecoration(
        color: YS.card,
        border: Border(bottom: BorderSide(color: YS.stroke))),
    child: Row(children: [
      Text('FINGERPRINT',
          style: YS.label(11, color: YS.amber, w: FontWeight.w800)
              .copyWith(letterSpacing: 1.5)),
      const Spacer(),
      // Auto-capture toggle
      GestureDetector(
        onTap: _toggleAutoCapture,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            color: _autoCapture ? YS.amberSoft : YS.cardAlt,
            border: Border.all(
                color: _autoCapture ? YS.amber : YS.stroke),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(
              _autoCapture ? Icons.auto_mode_rounded : Icons.touch_app_rounded,
              size: 12,
              color: _autoCapture ? YS.amberDeep : YS.inkLight,
            ),
            const SizedBox(width: 4),
            Text(_autoCapture ? 'AUTO' : 'MANUAL',
                style: YS.label(9,
                    color: _autoCapture ? YS.amberDeep : YS.inkLight,
                    w: FontWeight.w700)),
          ]),
        ),
      ),
      const SizedBox(width: 8),
      // Torch toggle
      GestureDetector(
        onTap: _live ? _toggleTorch : null,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: _torchOn ? YS.amber : YS.stroke),
            color: _torchOn ? YS.amberSoft : YS.cardAlt,
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(_torchOn ? Icons.flash_on : Icons.flash_off,
                size: 12,
                color: _torchOn ? YS.amberDeep : YS.inkLight),
            const SizedBox(width: 4),
            Text(_torchOn ? 'ON' : 'OFF',
                style: YS.label(9,
                    color: _torchOn ? YS.amberDeep : YS.inkLight,
                    w: FontWeight.w700)),
          ]),
        ),
      ),
    ]),
  );

  Widget _viewfinder() => AspectRatio(
    aspectRatio: 3 / 4,
    child: Container(
      color: const Color(0xFF1A1A1A),
      child: Stack(fit: StackFit.expand, children: [

        // Camera preview
        if (_live && _ctrl != null && _ctrl!.value.isInitialized)
          LayoutBuilder(
            builder: (ctx, c) => GestureDetector(
              onTapDown: (d) => _onTap(d, c),
              child: CameraPreview(_ctrl!),
            ),
          ),

        // Captured image preview
        if (_captured != null)
          Image.file(_captured!, fit: BoxFit.cover),

        // Idle state
        if (!_live && _captured == null)
          Container(
            color: const Color(0xFF1A1A1A),
            child: _initializing
                ? const Center(
                    child: CircularProgressIndicator(
                        color: YS.amber, strokeWidth: 2))
                : GestureDetector(
                    onTap: _startCamera,
                    child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Container(
                            width: 64, height: 64,
                            decoration: BoxDecoration(
                              color: YS.amber.withValues(alpha: 0.15),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.fingerprint,
                                size: 36, color: YS.amber),
                          ),
                          const SizedBox(height: 12),
                          Text('TAP TO START CAMERA',
                              style: YS.label(11,
                                  color: Colors.white54,
                                  w: FontWeight.w600)
                                  .copyWith(letterSpacing: 1.5)),
                        ]),
                  ),
          ),

        // Oval overlay
        CustomPaint(painter: _OverlayPainter(YS.amber)),

        // Scan line (only in auto mode)
        if (_live && _autoCapture)
          AnimatedBuilder(
            animation: _scanAnim,
            builder: (_, s) => Positioned(
              top: _scanAnim.value *
                  (MediaQuery.of(context).size.width * 4 / 3 * 0.6),
              left: MediaQuery.of(context).size.width * 0.18,
              right: MediaQuery.of(context).size.width * 0.18,
              child: Container(
                height: 2,
                decoration: const BoxDecoration(
                  gradient: LinearGradient(colors: [
                    Colors.transparent, YS.amber, Colors.transparent
                  ]),
                ),
              ),
            ),
          ),

        // Focus ring
        if (_focusPt != null)
          Positioned(
            left: _focusPt!.dx - 25, top: _focusPt!.dy - 25,
            child: Container(
              width: 50, height: 50,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(color: YS.amber, width: 2),
              ),
            ),
          ),

        // Flash white-out
        if (_flashing)
          Container(color: Colors.white.withValues(alpha: 0.8)),

        // ── Quality bar (auto mode only) ──────────────────────────────────
        if (_live && _autoCapture)
          Positioned(
            top: 12, left: 16, right: 16,
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                Text('QUALITY',
                    style: YS.label(9, color: Colors.white54, w: FontWeight.w700)
                        .copyWith(letterSpacing: 1.5)),
                // Pass counter dots
                Row(children: List.generate(_passesNeeded, (i) => Container(
                  width: 7, height: 7,
                  margin: const EdgeInsets.only(left: 4),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: i < _passCount
                        ? YS.amber
                        : Colors.white.withValues(alpha: 0.25),
                  ),
                ))),
              ]),
              const SizedBox(height: 4),
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: _qualityScore,
                  minHeight: 4,
                  backgroundColor: Colors.white.withValues(alpha: 0.15),
                  color: _qualityScore > 0.6
                      ? YS.amber
                      : _qualityScore > 0.3
                          ? Colors.orangeAccent
                          : Colors.redAccent,
                ),
              ),
            ]),
          ),

        // ── Guidance text ─────────────────────────────────────────────────
        Positioned(
          bottom: 14, left: 16, right: 16,
          child: Text(
            _captured != null
                ? '✓  Image captured'
                : _guidance,
            textAlign: TextAlign.center,
            style: YS.label(11,
                color: _captured != null ? YS.amber : _guidanceColor,
                w: FontWeight.w500),
          ),
        ),
      ]),
    ),
  );

  Widget _controls() => Container(
    color: YS.card,
    padding: const EdgeInsets.all(16),
    child: Column(children: [

      // Zoom slider
      if (_maxZoom > _minZoom && _live)
        Row(children: [
          const Icon(Icons.zoom_out, size: 14, color: YS.inkLight),
          Expanded(
            child: Slider(
              value: _zoom,
              min: _minZoom,
              max: _maxZoom.clamp(_minZoom, 5.0),
              activeColor: YS.amber,
              inactiveColor: YS.stroke,
              thumbColor: YS.amber,
              onChanged: (v) async {
                setState(() => _zoom = v);
                await _ctrl?.setZoomLevel(v);
              },
            ),
          ),
          const Icon(Icons.zoom_in, size: 14, color: YS.inkLight),
        ]),

      // Buttons
      if (!_live && _captured == null)
        _btn('Start Camera', _startCamera, true),

      if (_live && !_autoCapture)
        Row(children: [
          Expanded(child: _btn('Capture', _capture, true)),
          const SizedBox(width: 10),
          _btn('✕', _stopCamera, false, square: true),
        ]),

      if (_live && _autoCapture)
        Row(children: [
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 14),
              decoration: BoxDecoration(
                color: YS.amberSoft,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: YS.amber.withValues(alpha: 0.4)),
              ),
              child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                SizedBox(
                  width: 14, height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: YS.amberDeep,
                    value: _passCount > 0 ? _passCount / _passesNeeded : null,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  _passCount > 0
                      ? 'Hold still... $_passCount/$_passesNeeded'
                      : 'Scanning quality...',
                  style: YS.label(13, color: YS.amberDeep, w: FontWeight.w600),
                ),
              ]),
            ),
          ),
          const SizedBox(width: 10),
          _btn('✕', _stopCamera, false, square: true),
        ]),

      if (_captured != null)
        Row(children: [
          Expanded(child: _btn('Use This Image', _useImage, true)),
          const SizedBox(width: 10),
          Expanded(child: _btn('Retake', _retry, false)),
        ]),

      if (_error != null) ...[
        const SizedBox(height: 8),
        Text(_error!, style: const TextStyle(color: YS.red, fontSize: 11)),
      ],
    ]),
  );

  Widget _btn(String label, VoidCallback onTap, bool primary,
      {bool square = false}) =>
      GestureDetector(
        onTap: widget.disabled ? null : onTap,
        child: Container(
          width: square ? 48 : null,
          padding: EdgeInsets.symmetric(
              horizontal: square ? 0 : 16, vertical: 14),
          decoration: BoxDecoration(
            color: primary ? YS.amber : YS.cardAlt,
            borderRadius: BorderRadius.circular(12),
            border: primary ? null : Border.all(color: YS.stroke),
          ),
          child: Text(label,
              textAlign: TextAlign.center,
              style: YS.label(13,
                  color: primary ? Colors.black : YS.inkMid,
                  w: FontWeight.w700)),
        ),
      );
}

// ── Oval overlay painter ─────────────────────────────────────────────────────

class _OverlayPainter extends CustomPainter {
  final Color color;
  const _OverlayPainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2, cy = size.height / 2;
    final rx = size.width * 0.23, ry = size.height * 0.25;
    final oval = Rect.fromCenter(
        center: Offset(cx, cy), width: rx * 2, height: ry * 2);

    canvas.drawPath(
      Path()
        ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
        ..addOval(oval)
        ..fillType = PathFillType.evenOdd,
      Paint()..color = const Color(0xFF050505).withValues(alpha: 0.65),
    );
    canvas.drawOval(
        oval,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.5
          ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3));
    canvas.drawOval(
        oval,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.2);
  }

  @override
  bool shouldRepaint(_OverlayPainter old) => old.color != color;
}
