import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../api_client.dart';
import '../theme.dart';

const minReferencePhotos = 3;
const maxReferencePhotos = 10;

class EnrollScreen extends StatefulWidget {
  const EnrollScreen({super.key});

  @override
  State<EnrollScreen> createState() => _EnrollScreenState();
}

class _EnrollScreenState extends State<EnrollScreen> {
  final _api = ApiClient();
  final _picker = ImagePicker();
  final _guidanceController = TextEditingController();

  List<XFile> _files = [];
  bool _submitting = false;
  String? _error;
  EnrollResult? _result;

  Future<void> _pickImages() async {
    final picked = await _picker.pickMultiImage();
    if (picked.isEmpty || !mounted) return;
    setState(() {
      _files = picked;
      _error = null;
      _result = null;
    });
  }

  Future<void> _submit() async {
    final guidance = _guidanceController.text.trim();
    if (_files.length < minReferencePhotos) {
      setState(() => _error = 'Please choose at least $minReferencePhotos reference photos.');
      return;
    }
    if (_files.length > maxReferencePhotos) {
      setState(() => _error = 'Please choose at most $maxReferencePhotos reference photos.');
      return;
    }
    if (guidance.isEmpty) {
      setState(() => _error = 'Please describe what to do when this is matched.');
      return;
    }

    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      final result = await _api.enroll(files: _files, guidance: guidance);
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  void dispose() {
    _guidanceController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Set Target')),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          const Text(
            'Upload 3-10 example photos of the specific problem you want treated, and describe what to do about it. This replaces any previously set target.',
            style: TextStyle(color: muted),
          ),
          const SizedBox(height: 16),
          InkWell(
            onTap: _submitting ? null : _pickImages,
            borderRadius: BorderRadius.circular(12),
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: const Color(0xFF0F1B17),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: border, style: BorderStyle.solid),
              ),
              child: Column(
                children: [
                  const Icon(Icons.collections_outlined, size: 32, color: accent),
                  const SizedBox(height: 8),
                  Text(
                    _files.isEmpty
                        ? 'Tap to choose reference photos'
                        : '${_files.length} photo${_files.length == 1 ? '' : 's'} selected',
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 4),
                  const Text('png / jpg · 3-10 photos', style: TextStyle(color: muted, fontSize: 12)),
                ],
              ),
            ),
          ),
          if (_files.isNotEmpty) ...[
            const SizedBox(height: 12),
            SizedBox(
              height: 72,
              child: ListView.separated(
                scrollDirection: Axis.horizontal,
                itemCount: _files.length,
                separatorBuilder: (_, _) => const SizedBox(width: 8),
                itemBuilder: (_, i) => ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(File(_files[i].path), width: 72, height: 72, fit: BoxFit.cover),
                ),
              ),
            ),
          ],
          const SizedBox(height: 16),
          TextField(
            controller: _guidanceController,
            maxLines: 3,
            decoration: const InputDecoration(
              hintText: 'What should happen when this is matched? e.g. Spray organic neem oil mixture.',
            ),
          ),
          const SizedBox(height: 18),
          if (_error != null) _errorBanner(_error!),
          if (_result != null) _resultCard(_result!),
          const SizedBox(height: 18),
          ElevatedButton(
            onPressed: _submitting ? null : _submit,
            child: _submitting
                ? const SizedBox(
                    height: 18,
                    width: 18,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF06231A)),
                  )
                : const Text('Set Target'),
          ),
        ],
      ),
    );
  }

  Widget _errorBanner(String message) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: danger.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: danger.withValues(alpha: 0.35)),
      ),
      child: Text(message, style: const TextStyle(color: danger)),
    );
  }

  Widget _resultCard(EnrollResult result) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: card,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
            decoration: BoxDecoration(
              color: accent.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(999),
            ),
            child: const Text('Target set', style: TextStyle(color: accent, fontWeight: FontWeight.bold)),
          ),
          const SizedBox(height: 10),
          Text('Learned from ${result.numReferences} reference photos.', style: const TextStyle(color: muted)),
          const SizedBox(height: 6),
          Text(result.guidance),
        ],
      ),
    );
  }
}
