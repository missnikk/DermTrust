import 'dart:io';
import 'package:flutter/material.dart';
import '../models/diagnosis_result.dart';
import '../utils/app_theme.dart';
import '../widgets/confidence_indicator.dart';
import '../widgets/disease_card.dart';
import '../widgets/gradcam_viewer.dart';
import '../widgets/safety_banner.dart';

class ResultsScreen extends StatelessWidget {
  final DiagnosisResult result;

  const ResultsScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('Analysis Results'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded),
            tooltip: 'Analyze Another Image',
            onPressed: () => Navigator.of(context).popUntil((route) => route.isFirst),
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (result.isRecognized) ...[
              // Recognized Target Condition Path
              ConfidenceIndicator(
                confidence: result.confidence,
                entropy: result.entropy,
                isRecognized: true,
              ),

              const SizedBox(height: 16),

              // Grad-CAM Visual Heatmap
              GradCamViewer(
                originalImagePath: result.originalImagePath,
                gradCamBytes: result.gradCamOverlayBytes,
                diseaseName: result.predictedDisplayName,
              ),

              const SizedBox(height: 16),

              // Detailed Clinical Findings Card
              if (result.diseaseInfo != null)
                DiseaseCard(
                  info: result.diseaseInfo!,
                  acneSeverity: result.acneSeverity,
                ),

              const SizedBox(height: 16),

              // Probability Breakdown Across Target Classes
              _buildProbabilityBreakdown(result.classProbabilities),
            ] else ...[
              // Rejection / Non-Target Condition Path (Outcome 6)
              _buildRejectionBanner(context),

              const SizedBox(height: 16),

              // Image Preview
              Container(
                height: 240,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppTheme.dangerRose.withOpacity(0.5)),
                ),
                clipBehavior: Clip.antiAlias,
                child: Image.file(
                  File(result.originalImagePath),
                  fit: BoxFit.cover,
                ),
              ),

              const SizedBox(height: 16),

              if (result.classProbabilities.isNotEmpty)
                _buildProbabilityBreakdown(result.classProbabilities),
            ],

            const SizedBox(height: 16),
            const SafetyBanner(),
            const SizedBox(height: 24),

            // Analyze Another Button
            ElevatedButton.icon(
              onPressed: () => Navigator.of(context).popUntil((route) => route.isFirst),
              icon: const Icon(Icons.add_photo_alternate_outlined, size: 18),
              label: const Text('Analyze Another Image'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildRejectionBanner(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.dangerRose.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.dangerRose.withOpacity(0.5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.shield_outlined, color: AppTheme.dangerRose, size: 22),
              SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Outcome 6: Non-Target / Uncertain',
                  style: TextStyle(
                    color: AppTheme.dangerRose,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            result.rejectionReason ??
                'The system declined to force a classification because the lesion does not match the 5 supported categories with reliable statistical confidence.',
            style: const TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 13,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppTheme.surfaceElevated,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Row(
              children: [
                Icon(Icons.verified_user_outlined, color: AppTheme.accentCyan, size: 16),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'DWH Safety Guarantee: Rejection prevents unsafe diagnostic false positives on non-target diseases.',
                    style: TextStyle(color: AppTheme.textSecondary, fontSize: 11),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProbabilityBreakdown(Map<String, double> probs) {
    final sorted = probs.entries.toList()..sort((a, b) => b.value.compareTo(a.value));

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.surfaceBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'MODEL CONFIDENCE DISTRIBUTION',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              letterSpacing: 0.8,
              color: AppTheme.textMuted,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Uncalibrated model output across the 5 target classes (not diagnostic probabilities).',
            style: TextStyle(
              fontSize: 11,
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 10),
          ...sorted.map((e) {
            final pct = (e.value * 100).toStringAsFixed(1);
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        e.key.toUpperCase(),
                        style: const TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      Text(
                        '$pct%',
                        style: const TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: e.value.clamp(0.0, 1.0),
                      minHeight: 4,
                      backgroundColor: AppTheme.surfaceElevated,
                      valueColor: AlwaysStoppedAnimation<Color>(
                        e.value >= 0.60 ? AppTheme.primaryBlue : AppTheme.surfaceBorder,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }
}
