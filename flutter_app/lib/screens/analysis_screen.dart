import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/diagnosis_result.dart';
import '../services/inference_service.dart';
import '../utils/app_theme.dart';
import 'results_screen.dart';

class AnalysisScreen extends StatefulWidget {
  final File imageFile;

  const AnalysisScreen({super.key, required this.imageFile});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  String _currentStep = 'Preprocessing Image Tensor [1, 224, 224, 3]...';
  double _progress = 0.25;
  bool _hasError = false;
  String _errorMessage = '';

  @override
  void initState() {
    super.initState();
    _executePipeline();
  }

  Future<void> _executePipeline() async {
    try {
      final service = Provider.of<InferenceService>(context, listen: false);

      await Future.delayed(const Duration(milliseconds: 250));
      if (!mounted) return;
      setState(() {
        _currentStep = 'Executing MobileNetV2 Feature Extraction...';
        _progress = 0.50;
      });

      await Future.delayed(const Duration(milliseconds: 300));
      if (!mounted) return;
      setState(() {
        _currentStep = 'Evaluating DWH Rejection & Uncertainty Gates...';
        _progress = 0.75;
      });

      final result = await service.analyzeImage(widget.imageFile);

      if (!mounted) return;
      setState(() {
        _currentStep = 'Generating On-Device Grad-CAM Heatmap...';
        _progress = 1.0;
      });

      await Future.delayed(const Duration(milliseconds: 250));
      if (!mounted) return;

      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (_, __, ___) => ResultsScreen(result: result),
          transitionsBuilder: (_, anim, __, child) => FadeTransition(opacity: anim, child: child),
          transitionDuration: const Duration(milliseconds: 400),
        ),
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _hasError = true;
          _errorMessage = e.toString();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (!_hasError) ...[
                Container(
                  width: 80,
                  height: 80,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: AppTheme.surfaceElevated,
                    shape: BoxShape.circle,
                    border: Border.all(color: AppTheme.primaryBlue.withOpacity(0.4)),
                  ),
                  child: const CircularProgressIndicator(
                    strokeWidth: 3,
                    valueColor: AlwaysStoppedAnimation<Color>(AppTheme.accentCyan),
                  ),
                ),
                const SizedBox(height: 28),
                const Text(
                  'Local Deep Learning Inference',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _currentStep,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 13,
                    color: AppTheme.textSecondary,
                  ),
                ),
                const SizedBox(height: 24),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: _progress,
                    minHeight: 5,
                    backgroundColor: AppTheme.surfaceBorder,
                    valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.primaryBlue),
                  ),
                ),
              ] else ...[
                const Icon(Icons.error_outline, size: 48, color: AppTheme.dangerRose),
                const SizedBox(height: 16),
                const Text(
                  'Analysis Error',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _errorMessage,
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 13, color: AppTheme.textSecondary),
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Back to Image Preparation'),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
