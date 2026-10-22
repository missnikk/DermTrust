import 'package:flutter/material.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Text('About DermTrust'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header Card
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                color: AppTheme.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppTheme.primaryBlue.withOpacity(0.4)),
              ),
              child: Column(
                children: [
                  Container(
                    width: 56,
                    height: 56,
                    decoration: BoxDecoration(
                      color: AppTheme.primaryBlue.withOpacity(0.2),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(Icons.school_outlined, color: AppTheme.primaryBlue, size: 28),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    AppConstants.appName,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    AppConstants.appVersion,
                    style: TextStyle(
                      fontSize: 12,
                      color: AppTheme.accentCyan,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 18),

            _buildSection(
              title: 'Academic Context & Objectives',
              icon: Icons.biotech_outlined,
              content:
                  'DermTrust is developed as a final-year engineering major project addressing disparities in dermatological healthcare access. It explores how edge-deployed deep learning, digital image processing, and Explainable AI (XAI) can deliver reliable decision support in rural or network-deprived environments without cloud infrastructure.',
            ),

            const SizedBox(height: 14),

            _buildSection(
              title: 'Dermatological Workflow Hierarchy (DWH)',
              icon: Icons.account_tree_outlined,
              content:
                  'Rather than forcing every image into one of 5 diseases, DermTrust implements a multi-tier clinical safety pipeline:\n\n'
                  '• Tier 1 (Quality Validation): Rejects blurry, occluded, or low-resolution images via discrete 2D Laplacian sharpness evaluation.\n'
                  '• Tier 2 (Rejection Gate): Calculates predictive Shannon entropy and calibrated probability thresholds to identify non-target conditions.\n'
                  '• Tier 3 (Primary Classification): MobileNetV2 with spatial attention evaluates morphological features.\n'
                  '• Tier 4 (Grad-CAM Explainability): Computes localized visual activation maps verifying lesion attention.',
            ),

            const SizedBox(height: 14),

            _buildSection(
              title: 'Skin-Tone Fairness & Responsible AI',
              icon: Icons.balance_outlined,
              content:
                  'DermTrust audits performance across Fitzpatrick Skin Phototypes (Light: Types I–III vs Darker: Types IV–VI):\n\n'
                  '• Light Skin Accuracy: 79.59% (Macro-F1: 0.7286)\n'
                  '• Darker Skin Accuracy: 71.93% (Macro-F1: 0.6955)\n'
                  '• Disparity Gap |ΔF1|: 0.0332 (3.3%)\n\n'
                  'The evaluation confirms high diagnostic sensitivity across diverse skin phototypes without systemic underperformance on darker skin tones.',
            ),

            const SizedBox(height: 14),

            _buildSection(
              title: 'Datasets & Strict Licensing',
              icon: Icons.dataset_outlined,
              content:
                  'The core model is trained and evaluated using the verified Fitzpatrick 17K dataset under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) with zero data leakage across stratified train, validation, and untouched test splits.',
            ),

            const SizedBox(height: 14),

            _buildSection(
              title: 'Limitations & Future Work',
              icon: Icons.lightbulb_outline,
              content:
                  '• Decision Support Only: The app is not a certified diagnostic medical device.\n'
                  '• Clinical Variations: Rare subtypes and atypical presentations require in-person biopsy or dermoscopy.\n'
                  '• Additional Conditions: Future extensions will incorporate broader datasets for melasma, tinea, and pediatric conditions.',
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildSection({
    required String title,
    required IconData icon,
    required String content,
  }) {
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
          Row(
            children: [
              Icon(icon, color: AppTheme.primaryBlue, size: 18),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            content,
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 12,
              height: 1.45,
            ),
          ),
        ],
      ),
    );
  }
}
