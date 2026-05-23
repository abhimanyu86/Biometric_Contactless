import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme/app_theme.dart';

class FingerprintCameraWidget extends StatefulWidget {
  final void Function(File) onImageCaptured;
  final bool disabled;
  const FingerprintCameraWidget({super.key, required this.onImageCaptured, this.disabled = false});

  @override
  State<FingerprintCameraWidget> createState() => _FingerprintCameraWidgetState();
}

class _FingerprintCameraWidgetState extends State<FingerprintCameraWidget>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {
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

  late AnimationController _scanCtrl;
  late Animation<double> _scanAnim;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _scanCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 1800))
      ..repeat(reverse: true);
    _scanAnim = Tween<double>(begin: 0, end: 1).animate(
        CurvedAnimation(parent: _scanCtrl, curve: Curves.easeInOut));
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _focusTimer?.cancel();
    _scanCtrl.dispose();
    _ctrl?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.inactive) _stopCamera();
  }

  Future<void> _startCamera() async {
    if (widget.disabled || _initializing) return;
    setState(() { _initializing = true; _error = null; });
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) throw Exception('No camera found');
      final cam = cams.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      _ctrl = CameraController(cam, ResolutionPreset.max, enableAudio: false,
          imageFormatGroup: ImageFormatGroup.jpeg);
      await _ctrl!.initialize();
      await _ctrl!.setFocusMode(FocusMode.auto);
      await _ctrl!.setExposureMode(ExposureMode.auto);
      _minZoom = await _ctrl!.getMinZoomLevel();
      _maxZoom = await _ctrl!.getMaxZoomLevel();
      _zoom = _minZoom;
      try { await _ctrl!.setFlashMode(FlashMode.off); } catch (_) {}
      if (mounted) setState(() { _live = true; _initializing = false; });
    } catch (e) {
      if (mounted) setState(() { _error = e.toString(); _initializing = false; });
    }
  }

  Future<void> _stopCamera() async {
    if (_torchOn) { try { await _ctrl!.setFlashMode(FlashMode.off); } catch (_) {} }
    await _ctrl?.dispose(); _ctrl = null;
    if (mounted) setState(() { _live = false; _torchOn = false; _captured = null; });
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
    await Future.delayed(const Duration(milliseconds: 300));
    setState(() => _flashing = true);
    await Future.delayed(const Duration(milliseconds: 120));
    setState(() => _flashing = false);
    try {
      final xf = await _ctrl!.takePicture();
      if (mounted) setState(() { _captured = File(xf.path); _live = false; });
    } catch (e) {
      if (mounted) setState(() => _error = 'Capture failed: $e');
    }
  }

  void _useImage() { if (_captured != null) widget.onImageCaptured(_captured!); }

  Future<void> _retry() async {
    setState(() { _captured = null; _live = false; });
    await _startCamera();
  }

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
    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
    decoration: const BoxDecoration(
      color: YS.card,
      border: Border(bottom: BorderSide(color: YS.stroke))),
    child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
      Text('FINGERPRINT CAPTURE', style: YS.label(11, color: YS.amber, w: FontWeight.w800).copyWith(letterSpacing: 1.5)),
      GestureDetector(
        onTap: _live ? _toggleTorch : null,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: _torchOn ? YS.amber : YS.stroke),
            color: _torchOn ? YS.amberSoft : YS.cardAlt,
          ),
          child: Row(children: [
            Icon(_torchOn ? Icons.flash_on : Icons.flash_off, size: 13,
                color: _torchOn ? YS.amberDeep : YS.inkLight),
            const SizedBox(width: 4),
            Text(_torchOn ? 'ON' : 'OFF', style: YS.label(9, color: _torchOn ? YS.amberDeep : YS.inkLight, w: FontWeight.w700)),
          ]),
        ),
      ),
    ]),
  );

  Widget _viewfinder() => AspectRatio(
    aspectRatio: 3/4,
    child: Container(
      color: const Color(0xFF1A1A1A),
      child: Stack(fit: StackFit.expand, children: [
        if (_live && _ctrl != null && _ctrl!.value.isInitialized)
          LayoutBuilder(builder: (ctx, c) => GestureDetector(
            onTapDown: (d) => _onTap(d, c), child: CameraPreview(_ctrl!))),
        if (_captured != null) Image.file(_captured!, fit: BoxFit.cover),
        if (!_live && _captured == null)
          Container(
            color: const Color(0xFF1A1A1A),
            child: _initializing
                ? const Center(child: CircularProgressIndicator(color: YS.amber, strokeWidth: 2))
                : GestureDetector(
                    onTap: _startCamera,
                    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                      Container(width: 64, height: 64,
                        decoration: BoxDecoration(color: YS.amber.withOpacity(0.15),
                            shape: BoxShape.circle),
                        child: const Icon(Icons.fingerprint, size: 36, color: YS.amber)),
                      const SizedBox(height: 12),
                      Text('TAP TO START CAMERA', style: YS.label(11, color: Colors.white54, w: FontWeight.w600).copyWith(letterSpacing: 1.5)),
                    ]),
                  ),
          ),
        CustomPaint(painter: _OverlayPainter(YS.amber)),
        if (_live) AnimatedBuilder(
          animation: _scanAnim,
          builder: (_, __) => Positioned(
            top: _scanAnim.value * (MediaQuery.of(context).size.width * 4/3 * 0.6),
            left: MediaQuery.of(context).size.width * 0.18,
            right: MediaQuery.of(context).size.width * 0.18,
            child: Container(height: 2, decoration: BoxDecoration(
              gradient: LinearGradient(colors: [
                Colors.transparent, YS.amber, Colors.transparent]))),
          ),
        ),
        if (_focusPt != null) Positioned(
          left: _focusPt!.dx - 25, top: _focusPt!.dy - 25,
          child: Container(width: 50, height: 50, decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: YS.amber, width: 2))),
        ),
        if (_flashing) Container(color: Colors.white.withOpacity(0.8)),
        Positioned(bottom: 14, left: 0, right: 0,
          child: Text(
            _captured != null ? '✓  Image captured' : 'Place finger inside the oval',
            textAlign: TextAlign.center,
            style: YS.label(11, color: _captured != null ? YS.amber : Colors.white60, w: FontWeight.w500),
          ),
        ),
      ]),
    ),
  );

  Widget _controls() => Container(
    color: YS.card,
    padding: const EdgeInsets.all(16),
    child: Column(children: [
      if (_maxZoom > _minZoom && _live) Row(children: [
        const Icon(Icons.zoom_out, size: 14, color: YS.inkLight),
        Expanded(child: Slider(
          value: _zoom, min: _minZoom, max: _maxZoom.clamp(_minZoom, 5.0),
          activeColor: YS.amber, inactiveColor: YS.stroke,
          thumbColor: YS.amber,
          onChanged: (v) async { setState(() => _zoom = v); await _ctrl?.setZoomLevel(v); },
        )),
        const Icon(Icons.zoom_in, size: 14, color: YS.inkLight),
      ]),
      if (!_live && _captured == null) _btn('Start Camera', _startCamera, true),
      if (_live) Row(children: [
        Expanded(child: _btn('Capture', _capture, true)),
        const SizedBox(width: 10),
        _btn('✕', _stopCamera, false, square: true),
      ]),
      if (_captured != null) Row(children: [
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

  Widget _btn(String label, VoidCallback onTap, bool primary, {bool square = false}) =>
    GestureDetector(
      onTap: widget.disabled ? null : onTap,
      child: Container(
        width: square ? 48 : null,
        padding: EdgeInsets.symmetric(horizontal: square ? 0 : 16, vertical: 14),
        decoration: BoxDecoration(
          color: primary ? YS.amber : YS.cardAlt,
          borderRadius: BorderRadius.circular(12),
          border: primary ? null : Border.all(color: YS.stroke),
        ),
        child: Text(label, textAlign: TextAlign.center,
          style: YS.label(13, color: primary ? Colors.black : YS.inkMid, w: FontWeight.w700)),
      ),
    );
}

class _OverlayPainter extends CustomPainter {
  final Color color;
  const _OverlayPainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width/2, cy = size.height/2;
    final rx = size.width*0.23, ry = size.height*0.25;
    final oval = Rect.fromCenter(center: Offset(cx,cy), width: rx*2, height: ry*2);
    canvas.drawPath(
      Path()..addRect(Rect.fromLTWH(0,0,size.width,size.height))..addOval(oval)
        ..fillType = PathFillType.evenOdd,
      Paint()..color = const Color(0xFF050505).withOpacity(0.65),
    );
    canvas.drawOval(oval, Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = 1.5
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3));
    canvas.drawOval(oval, Paint()..color = color..style = PaintingStyle.stroke..strokeWidth = 1.2);
  }

  @override
  bool shouldRepaint(_OverlayPainter old) => old.color != color;
}
