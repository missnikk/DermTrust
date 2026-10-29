import 'package:flutter/material.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 2,
      child: Scaffold(
        backgroundColor: AppTheme.background,
        appBar: AppBar(
          title: const Text('About & Clinical Safety'),
          bottom: const TabBar(
            indicatorColor: AppTheme.primaryBlue,
            labelColor: AppTheme.textPrimary,
            unselectedLabelColor: AppTheme.textSecondary,
            labelStyle: TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
            tabs: [
              Tab(text: 'Research Overview'),
              Tab(text: 'Safety & Privacy'),
            ],
          ),
        ),
        body: TabBarView(
          children: [_buildOverviewTab(), _buildSafetyAndPrivacyTab()],
        ),
      ),
    );
  }

  Widget _buildOverviewTab() {
    return SingleChildScrollView(
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
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: AppTheme.primaryBlue.withOpacity(0.15),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.school_outlined,
                    color: AppTheme.primaryBlue,
                    size: 26,
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  AppConstants.appName,
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: AppTheme.textPrimary,
                    letterSpacing: -0.2,
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

          const SizedBox(height: 16),

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
    );
  }

  Widget _buildSafetyAndPrivacyTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Zero-Network Banner
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.primaryBlue.withOpacity(0.12),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.primaryBlue.withOpacity(0.4)),
            ),
            child: const Row(
              children: [
                Icon(
                  Icons.lock_outline_rounded,
                  color: AppTheme.primaryBlue,
                  size: 28,
                ),
                SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Zero-Network Offline Guarantee',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 15,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        'DermTrust operates with 100% local edge inference. No user photographs or clinical data ever leave your physical device.',
                        style: TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 12,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          _buildSection(
            title: 'Volatile In-Memory Processing',
            icon: Icons.memory_outlined,
            content:
                'Photographs selected from the gallery or captured by the camera are converted into numerical RGB tensors in volatile memory (RAM). Once the session ends, the temporary arrays are discarded by the operating system garbage collector.',
          ),

          const SizedBox(height: 14),

          _buildSection(
            title: 'No Remote Databases or Telemetry',
            icon: Icons.cloud_off_outlined,
            content:
                'The application does not include analytics trackers, user registration systems, cloud synchronization, or remote APIs. It functions completely without internet, cellular data, or Wi-Fi.',
          ),

          const SizedBox(height: 14),

          _buildSection(
            title: 'No Racial or Biometric Profiling',
            icon: Icons.fingerprint_outlined,
            content:
                'The Fitzpatrick phototyping framework evaluates optical ultraviolet erythema and melanin response. The application does not infer, estimate, or store user race, ethnicity, or personal biometric identities.',
          ),

          const SizedBox(height: 14),

          _buildSection(
            title: 'Clinical Safety Boundaries',
            icon: Icons.medical_services_outlined,
            content:
                'DermTrust will never prescribe medications, recommend prescription dosages, or advise you to avoid medical evaluation. If you suspect an adverse skin condition, always schedule an examination with a board-certified dermatologist.',
          ),

          const SizedBox(height: 14),

          _buildSection(
            title: 'Mandatory Clinical Disclaimer',
            icon: Icons.shield_outlined,
            content: AppConstants.medicalDisclaimer,
          ),

          const SizedBox(height: 24),
        ],
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
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    color: AppTheme.textPrimary,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
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
