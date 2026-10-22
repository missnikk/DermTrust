import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/disease_info.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';
import '../widgets/safety_banner.dart';
import 'about_screen.dart';
import 'image_prep_screen.dart';
import 'privacy_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Future<void> _pickImage(BuildContext context, ImageSource source) async {
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(
        source: source,
        maxWidth: 1600,
        maxHeight: 1600,
        imageQuality: 90,
      );

      if (picked != null && context.mounted) {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => ImagePrepScreen(imageFile: File(picked.path)),
          ),
        );
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to acquire image: $e'),
            backgroundColor: AppTheme.dangerRose,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.health_and_safety_rounded, color: AppTheme.primaryBlue, size: 20),
            SizedBox(width: 8),
            Text(AppConstants.appName),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline, size: 20),
            tooltip: 'About DermTrust',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const AboutScreen()),
              );
            },
          ),
          IconButton(
            icon: const Icon(Icons.privacy_tip_outlined, size: 20),
            tooltip: 'Privacy & Offline Guarantee',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const PrivacyScreen()),
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Offline Status Indicator Pill
            Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: AppTheme.successGreen.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppTheme.successGreen.withOpacity(0.3)),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.wifi_off_rounded, color: AppTheme.successGreen, size: 14),
                    SizedBox(width: 6),
                    Text(
                      '100% Offline Clinical Runtime Active',
                      style: TextStyle(
                        color: AppTheme.successGreen,
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 18),

            // Main Hero Card
            Container(
              padding: const EdgeInsets.all(18),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    AppTheme.surface,
                    const Color(0xFF1E293B).withOpacity(0.7),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppTheme.primaryBlue.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Clinical Skin Lesion Analysis',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'Evaluate skin images for 5 target conditions with on-device Grad-CAM visual explanations and responsible non-target rejection.',
                    style: TextStyle(
                      fontSize: 13,
                      color: AppTheme.textSecondary,
                      height: 1.45,
                    ),
                  ),
                  const SizedBox(height: 18),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () => _pickImage(context, ImageSource.gallery),
                          icon: const Icon(Icons.photo_library_outlined, size: 18),
                          label: const Text('From Gallery'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () => _pickImage(context, ImageSource.camera),
                          icon: const Icon(Icons.camera_alt_outlined, size: 18),
                          label: const Text('Take Photo'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 22),

            // Supported Target Diseases Section
            const Text(
              'TARGET CONDITIONS & REJECTION LOGIC',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.bold,
                letterSpacing: 0.8,
                color: AppTheme.textMuted,
              ),
            ),
            const SizedBox(height: 10),

            ...AppConstants.targetClasses.map((clsId) {
              final info = DiseaseInfo.getById(clsId);
              if (info == null) return const SizedBox.shrink();
              return Container(
                margin: const EdgeInsets.only(bottom: 8),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                decoration: BoxDecoration(
                  color: AppTheme.surface,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: AppTheme.surfaceBorder),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                        color: AppTheme.accentCyan,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            info.displayName,
                            style: const TextStyle(
                              color: AppTheme.textPrimary,
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            info.category,
                            style: const TextStyle(
                              color: AppTheme.textMuted,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const Icon(Icons.chevron_right, color: AppTheme.textMuted, size: 18),
                  ],
                ),
              );
            }),

            // Rejection / Uncertainty Outcome 6 Card
            Container(
              margin: const EdgeInsets.only(bottom: 8),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(
                color: AppTheme.surfaceElevated,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppTheme.warningAmber.withOpacity(0.3)),
              ),
              child: const Row(
                children: [
                  Icon(Icons.shield_outlined, color: AppTheme.warningAmber, size: 16),
                  SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Outcome 6: Rejection / Non-Target',
                          style: TextStyle(
                            color: AppTheme.warningAmber,
                            fontSize: 13,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          'Rejects out-of-distribution rashes, non-skin images, and ambiguous inputs.',
                          style: TextStyle(
                            color: AppTheme.textSecondary,
                            fontSize: 11,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),
            const SafetyBanner(),
          ],
        ),
      ),
    );
  }
}
