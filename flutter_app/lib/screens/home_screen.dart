import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../models/disease_info.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';
import 'about_screen.dart';
import 'history_screen.dart';
import 'image_prep_screen.dart';

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
        backgroundColor: Colors.transparent,
        elevation: 0,
        titleSpacing: 20,
        centerTitle: false,
        title: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                color: AppTheme.primaryBlue.withOpacity(0.15),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(
                  color: AppTheme.primaryBlue.withOpacity(0.35),
                ),
              ),
              child: const Icon(
                Icons.health_and_safety_rounded,
                color: AppTheme.primaryBlue,
                size: 20,
              ),
            ),
            const SizedBox(width: 10),
            const Text(
              AppConstants.appName,
              style: TextStyle(
                color: AppTheme.textPrimary,
                fontSize: 20,
                fontWeight: FontWeight.w700,
                letterSpacing: -0.3,
              ),
            ),
          ],
        ),
        actions: [
          // Single consolidated Information & Safety button
          IconButton(
            icon: const Icon(
              Icons.info_outline_rounded,
              color: AppTheme.textPrimary,
              size: 22,
            ),
            tooltip: 'About DermTrust & Safety Policy',
            onPressed: () {
              Navigator.of(
                context,
              ).push(MaterialPageRoute(builder: (_) => const AboutScreen()));
            },
          ),
          const SizedBox(width: 12),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 1. Brief Research Project Introduction
              _buildIntroductionHeader(),

              const SizedBox(height: 18),

              // 2. Compact Target Disease Chips
              _buildTargetConditionsSection(),

              const SizedBox(height: 20),

              // 3. Primary Image Selection Section (Take Photo & Choose from Gallery)
              _buildImageSelectionSection(context),

              const SizedBox(height: 20),

              // 4. Three Key Feature Claims (Bias Mitigation, Offline Analysis, Explainable AI)
              _buildKeyFeaturesSection(),

              const SizedBox(height: 18),

              // 5. History Navigation Button
              _buildHistoryButton(context),

              const SizedBox(height: 18),

              // 6. Clinical Research Disclaimer
              _buildDisclaimer(),

              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }

  // 1. Brief Research Project Introduction
  Widget _buildIntroductionHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
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
              Container(
                width: 7,
                height: 7,
                decoration: const BoxDecoration(
                  color: AppTheme.accentCyan,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              const Text(
                'ACADEMIC RESEARCH INITIATIVE',
                style: TextStyle(
                  color: AppTheme.accentCyan,
                  fontSize: 10.5,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.8,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          const Text(
            'DermTrust investigates edge-deployed deep learning and visual explainability for dermatological lesion assessment without cloud dependence.',
            style: TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 12.5,
              height: 1.45,
            ),
          ),
        ],
      ),
    );
  }

  // 2. Compact Target Disease Chips
  Widget _buildTargetConditionsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'SUPPORTED CONDITIONS',
              style: TextStyle(
                color: AppTheme.textMuted,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.8,
              ),
            ),
            Text(
              '5 Target Classes',
              style: TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 11,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children:
              AppConstants.targetClasses.map((clsId) {
                final info = DiseaseInfo.getById(clsId);
                final displayName = info?.displayName ?? clsId;
                return Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: AppTheme.surfaceElevated,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: AppTheme.surfaceBorder),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 5,
                        height: 5,
                        decoration: const BoxDecoration(
                          color: AppTheme.primaryBlue,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 6),
                      Text(
                        displayName,
                        style: const TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                );
              }).toList(),
        ),
      ],
    );
  }

  // 3. Primary Image Selection Section
  Widget _buildImageSelectionSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Action 1: Take Photo (Primary Accent)
        ElevatedButton(
          onPressed: () => _pickImage(context, ImageSource.camera),
          style: ElevatedButton.styleFrom(
            backgroundColor: AppTheme.primaryBlue,
            foregroundColor: Colors.white,
            elevation: 1.5,
            padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 18),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          child: const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.camera_alt_rounded, size: 20, color: Colors.white),
              SizedBox(width: 10),
              Text(
                'Take Photo',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.2,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 10),

        // Action 2: Choose from Gallery (Harmonious Blue Accent)
        OutlinedButton(
          onPressed: () => _pickImage(context, ImageSource.gallery),
          style: OutlinedButton.styleFrom(
            backgroundColor: AppTheme.galleryBlueSurface,
            foregroundColor: AppTheme.textPrimary,
            side: const BorderSide(
              color: AppTheme.galleryBlueBorder,
              width: 1.2,
            ),
            padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 18),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
          child: const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.photo_library_outlined,
                size: 20,
                color: AppTheme.accentBlueLight,
              ),
              SizedBox(width: 10),
              Text(
                'Choose from Gallery',
                style: TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.2,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // 4. Three Key Feature Claims
  Widget _buildKeyFeaturesSection() {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.surfaceBorder),
      ),
      child: Column(
        children: [
          _buildFeatureRow(
            title: 'Bias Mitigation',
            description:
                'Works toward more reliable performance across different skin tones.',
            icon: Icons.balance_rounded,
          ),
          Divider(
            color: AppTheme.surfaceBorder.withOpacity(0.6),
            height: 1,
            indent: 16,
            endIndent: 16,
          ),
          _buildFeatureRow(
            title: 'Offline Analysis',
            description:
                'Processes images on your device without requiring an internet connection for inference.',
            icon: Icons.wifi_off_rounded,
          ),
          Divider(
            color: AppTheme.surfaceBorder.withOpacity(0.6),
            height: 1,
            indent: 16,
            endIndent: 16,
          ),
          _buildFeatureRow(
            title: 'Explainable AI',
            description:
                'Highlights image regions that help show what influenced the model\'s prediction.',
            icon: Icons.visibility_outlined,
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureRow({
    required String title,
    required String description,
    required IconData icon,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: AppTheme.textPrimary,
                    fontSize: 13.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  description,
                  style: const TextStyle(
                    color: AppTheme.textSecondary,
                    fontSize: 11.5,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 14),
          Icon(icon, color: AppTheme.primaryBlue, size: 20),
        ],
      ),
    );
  }

  // 5. History Navigation Button
  Widget _buildHistoryButton(BuildContext context) {
    return InkWell(
      onTap: () {
        Navigator.of(
          context,
        ).push(MaterialPageRoute(builder: (_) => const HistoryScreen()));
      },
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
        decoration: BoxDecoration(
          color: AppTheme.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.surfaceBorder),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(7),
              decoration: BoxDecoration(
                color: AppTheme.primaryBlue.withOpacity(0.12),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(
                Icons.history_rounded,
                color: AppTheme.primaryBlue,
                size: 18,
              ),
            ),
            const SizedBox(width: 12),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'History',
                    style: TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 13.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  SizedBox(height: 1),
                  Text(
                    'Review past analyses and diagnostic logs',
                    style: TextStyle(color: AppTheme.textMuted, fontSize: 11),
                  ),
                ],
              ),
            ),
            const Icon(
              Icons.chevron_right_rounded,
              color: AppTheme.textMuted,
              size: 20,
            ),
          ],
        ),
      ),
    );
  }

  // 6. Clinical Research Disclaimer
  Widget _buildDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surface.withOpacity(0.5),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.surfaceBorder.withOpacity(0.6)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.info_outline_rounded,
            color: AppTheme.textMuted,
            size: 15,
          ),
          const SizedBox(width: 9),
          Expanded(
            child: Text(
              AppConstants.medicalDisclaimer,
              style: const TextStyle(
                color: AppTheme.textMuted,
                fontSize: 10.5,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
