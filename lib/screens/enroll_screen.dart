import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import '../theme/app_theme.dart';
import '../widgets/fingerprint_camera_widget.dart';
import '../services/api_service.dart';

class EnrollScreen extends StatefulWidget {
  const EnrollScreen({super.key});
  @override
  State<EnrollScreen> createState() => _EnrollScreenState();
}

class _EnrollScreenState extends State<EnrollScreen> {
  final _nameCtrl  = TextEditingController();
  final _uidCtrl   = TextEditingController();
  final _batchCtrl = TextEditingController();
  File? _image;
  bool _loading = false;
  Map<String, dynamic>? _result;

  @override
  void dispose() { _nameCtrl.dispose(); _uidCtrl.dispose(); _batchCtrl.dispose(); super.dispose(); }

  Future<void> _enroll() async {
    if (_nameCtrl.text.trim().isEmpty || _uidCtrl.text.trim().isEmpty || _batchCtrl.text.trim().isEmpty) {
      _snack('Fill all fields'); return;
    }
    if (_image == null) { _snack('Capture fingerprint first'); return; }
    setState(() { _loading = true; _result = null; });
    try {
      final r = await ApiService.enroll(
        name: _nameCtrl.text.trim(), uid: _uidCtrl.text.trim(),
        batch: _batchCtrl.text.trim(), image: _image!);
      setState(() { _result = r; _loading = false; });
      if (r['success'] == true) {
        _nameCtrl.clear(); _uidCtrl.clear(); _batchCtrl.clear();
        setState(() => _image = null);
      }
    } catch (e) {
      setState(() { _result = {'success': false, 'error': e.toString()}; _loading = false; });
    }
  }

  void _snack(String msg) => ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: YS.red));

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: YS.bg,
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back_rounded), onPressed: () => context.go('/')),
        title: Text('Enroll User', style: YS.label(17, w: FontWeight.w700)),
        bottom: PreferredSize(preferredSize: const Size.fromHeight(1),
            child: Container(height: 1, color: YS.stroke)),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          _badge('FINGERPRINT ENROLLMENT', YS.amber, YS.amberSoft),
          const SizedBox(height: 20),
          _field(_nameCtrl,  'Full Name',     'e.g. Priya Sharma',   Icons.person_outline_rounded),
          const SizedBox(height: 12),
          _field(_uidCtrl,   'Aadhaar UID',   'e.g. 1234-5678-9012', Icons.badge_outlined),
          const SizedBox(height: 12),
          _field(_batchCtrl, 'Batch / Group', 'e.g. BATCH_A',        Icons.group_outlined),
          const SizedBox(height: 24),
          Text('FINGERPRINT CAPTURE', style: YS.label(11, color: YS.inkLight, w: FontWeight.w700)
              .copyWith(letterSpacing: 1.8)),
          const SizedBox(height: 10),
          FingerprintCameraWidget(
            onImageCaptured: (f) => setState(() { _image = f; _result = null; }),
            disabled: _loading),
          const SizedBox(height: 20),
          if (_result != null) _resultCard(_result!),
          if (_result != null) const SizedBox(height: 16),
          ElevatedButton(
            onPressed: _loading ? null : _enroll,
            child: _loading
                ? const SizedBox(width: 18, height: 18,
                    child: CircularProgressIndicator(color: Colors.black, strokeWidth: 2))
                : const Text('Enroll User'),
          ),
          const SizedBox(height: 40),
        ]),
      ),
    );
  }

  Widget _field(TextEditingController c, String label, String hint, IconData icon) =>
    TextField(controller: c,
      style: YS.label(14),
      decoration: InputDecoration(labelText: label, hintText: hint,
          prefixIcon: Icon(icon, color: YS.inkLight, size: 18)));

  Widget _badge(String label, Color color, Color bg) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
    decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3))),
    child: Text(label, style: YS.label(10, color: color, w: FontWeight.w700)
        .copyWith(letterSpacing: 1.5)),
  );

  Widget _resultCard(Map<String, dynamic> r) {
    final ok = r['success'] == true;
    final color = ok ? YS.green : YS.red;
    final bg    = ok ? YS.greenBg : YS.redBg;
    String msg  = ok ? 'Enrolled successfully!' : (r['error'] ?? r['guidance'] ?? 'Failed');
    if (r['quality_failed'] == true) msg = r['guidance'] ?? 'Quality check failed';
    if (r['spoof_detected'] == true) msg = 'Spoof detected — use a live finger';
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(14),
          border: Border.all(color: color.withValues(alpha: 0.3))),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(ok ? Icons.check_circle_rounded : Icons.error_rounded, color: color, size: 18),
          const SizedBox(width: 8),
          Expanded(child: Text(msg, style: YS.label(13, color: color, w: FontWeight.w600))),
        ]),
        if (ok) ...[
          const SizedBox(height: 10),
          _row('Minutiae', '${r['minutiae_count']}'),
          _row('Liveness', r['liveness']?['is_live'] == true ? 'Live ✓' : 'Spoof'),
          _row('Detection', '${((r['detection_conf'] ?? 0) * 100).toStringAsFixed(1)}%'),
        ],
        if (r['issues'] != null && (r['issues'] as List).isNotEmpty) ...[
          const SizedBox(height: 8),
          ...(r['issues'] as List).map((i) => Text('• $i', style: YS.label(11, color: YS.orange))),
        ],
      ]),
    );
  }

  Widget _row(String k, String v) => Padding(
    padding: const EdgeInsets.only(bottom: 4),
    child: Row(children: [
      Text('$k: ', style: YS.label(12, color: YS.inkMid)),
      Text(v, style: YS.label(12, w: FontWeight.w700)),
    ]),
  );
}
