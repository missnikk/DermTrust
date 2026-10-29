import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class PrivacyScreen extends StatelessWidget {
  const PrivacyScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(title: const Text('Privacy & Safety Policy')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.successGreen.withOpacity(0.12),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                  color: AppTheme.successGreen.withOpacity(0.4),
                ),
              ),
              child: const Row(
                children: [
                  Icon(
                    Icons.lock_outline_rounded,
                    color: AppTheme.successGreen,
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

            const SizedBox(height: 18),

            _buildCard(
              title: 'Volatile In-Memory Processing',
              icon: Icons.memory_outlined,
              description:
                  'Photographs selected from the gallery or captured by the camera are converted into numerical RGB tensors in volatile memory (RAM). Once the session ends, the temporary arrays are discarded by the operating system garbage collector.',
            ),

            const SizedBox(height: 12),

            _buildCard(
              title: 'No Remote Databases or Telemetry',
              icon: Icons.cloud_off_outlined,
              description:
                  'The application does not include analytics trackers, user registration systems, cloud synchronization, or remote APIs. It functions completely without internet, cellular data, or Wi-Fi.',
            ),

            const SizedBox(height: 12),

            _buildCard(
              title: 'No Racial or Biometric Profiling',
              icon: Icons.fingerprint_outlined,
              description:
                  'The Fitzpatrick phototyping framework evaluates optical ultraviolet erythema and melanin response. The application does not infer, estimate, or store user race, ethnicity, or personal biometric identities.',
            ),

            const SizedBox(height: 12),

            _buildCard(
              title: 'Clinical Safety Boundaries',
              icon: Icons.medical_services_outlined,
              description:
                  'DermTrust will never prescribe medications, recommend prescription dosages, or advise you to avoid medical evaluation. If you suspect an adverse skin condition, always schedule an examination with a board-certified dermatologist.',
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildCard({
    required String title,
    required IconData icon,
    required String description,
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
              Icon(icon, color: AppTheme.accentCyan, size: 18),
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
          const SizedBox(height: 8),
          Text(
            description,
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
