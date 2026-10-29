import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/image_processor.dart';
import '../utils/app_theme.dart';
import 'analysis_screen.dart';

class ImagePrepScreen extends StatefulWidget {
  final File imageFile;

  const ImagePrepScreen({super.key, required this.imageFile});

  @override
  State<ImagePrepScreen> createState() => _ImagePrepScreenState();
}

class _ImagePrepScreenState extends State<ImagePrepScreen> {
  late File _currentFile;
  bool _isValidating = true;
  ImageValidationResult? _validationResult;

  @override
  void initState() {
    super.initState();
    _currentFile = widget.imageFile;
    _runValidation();
  }

  Future<void> _runValidation() async {
    setState(() => _isValidating = true);
    final result = await ImageProcessor.validateImage(_currentFile);
    if (mounted) {
      setState(() {
        _validationResult = result;
        _isValidating = false;
      });
    }
  }

  Future<void> _replaceImage(ImageSource source) async {
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(
        source: source,
        maxWidth: 1600,
        maxHeight: 1600,
        imageQuality: 90,
      );
      if (picked != null) {
        setState(() {
          _currentFile = File(picked.path);
        });
        _runValidation();
      }
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final isValid = _validationResult?.isValid ?? false;

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(title: const Text('Image Preparation')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Container
            Container(
              height: 280,
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color:
                      isValid
                          ? AppTheme.primaryBlue.withOpacity(0.5)
                          : AppTheme.dangerRose,
                  width: 2,
                ),
              ),
              clipBehavior: Clip.antiAlias,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(_currentFile, fit: BoxFit.cover),
                  if (_isValidating)
                    Container(
                      color: Colors.black54,
                      child: const Center(child: CircularProgressIndicator()),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // Replace / Retake Button Row
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextButton.icon(
                  onPressed: () => _replaceImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined, size: 16),
                  label: const Text('Change Photo'),
                ),
                const SizedBox(width: 16),
                TextButton.icon(
                  onPressed: () => _replaceImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt_outlined, size: 16),
                  label: const Text('Retake'),
                ),
              ],
            ),

            const SizedBox(height: 8),

            // Validation Status Box
            if (!_isValidating && _validationResult != null)
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color:
                      isValid
                          ? AppTheme.successGreen.withOpacity(0.12)
                          : AppTheme.dangerRose.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color:
                        isValid
                            ? AppTheme.successGreen.withOpacity(0.4)
                            : AppTheme.dangerRose.withOpacity(0.4),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      isValid
                          ? Icons.check_circle_outline
                          : Icons.error_outline,
                      color:
                          isValid ? AppTheme.successGreen : AppTheme.dangerRose,
                      size: 20,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        isValid
                            ? 'Image quality verified (Sharpness: ${_validationResult!.blurScore.toStringAsFixed(1)})'
                            : _validationResult!.errorMessage ??
                                'Image validation failed',
                        style: TextStyle(
                          color:
                              isValid
                                  ? AppTheme.successGreen
                                  : AppTheme.dangerRose,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

            const SizedBox(height: 18),

            // Clinical Capture Checklist
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppTheme.surface,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: AppTheme.surfaceBorder),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'CAPTURE QUALITY GUIDELINES',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 0.8,
                      color: AppTheme.textMuted,
                    ),
                  ),
                  SizedBox(height: 8),
                  _GuidelineItem(
                    text:
                        'Uniform, diffuse lighting without direct flash glare.',
                  ),
                  _GuidelineItem(
                    text:
                        'Lesion in sharp focus with adjacent normal skin for contrast.',
                  ),
                  _GuidelineItem(
                    text:
                        'Avoid beauty filters, extreme zoom, or digital artifacts.',
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // Analyze Action Button
            ElevatedButton(
              onPressed:
                  (!isValid || _isValidating)
                      ? null
                      : () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder:
                                (_) => AnalysisScreen(imageFile: _currentFile),
                          ),
                        );
                      },
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
              child: const Text('Start Diagnostic Analysis'),
            ),
          ],
        ),
      ),
    );
  }
}

class _GuidelineItem extends StatelessWidget {
  final String text;

  const _GuidelineItem({required this.text});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.check, size: 14, color: AppTheme.accentCyan),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
