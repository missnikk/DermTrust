import 'package:flutter/material.dart';
import '../utils/app_theme.dart';
import '../utils/constants.dart';

class SafetyBanner extends StatelessWidget {
  final bool compact;

  const SafetyBanner({super.key, this.compact = false});

  @override
  Widget build(BuildContext context) {
    if (compact) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: AppTheme.surfaceElevated,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: AppTheme.surfaceBorder),
        ),
        child: const Row(
          children: [
            Icon(Icons.shield_outlined, color: AppTheme.accentCyan, size: 16),
            SizedBox(width: 8),
            Expanded(
              child: Text(
                '100% Offline & Private: No images leave this device.',
                style: TextStyle(color: AppTheme.textSecondary, fontSize: 11),
              ),
            ),
          ],
        ),
      );
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E1B4B).withOpacity(0.4),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0xFF4338CA).withOpacity(0.5)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.info_outline, color: Color(0xFF818CF8), size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Clinical Decision Support Disclaimer',
                  style: TextStyle(
                    color: Color(0xFFC7D2FE),
                    fontWeight: FontWeight.w600,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  AppConstants.medicalDisclaimer,
                  style: const TextStyle(
                    color: Color(0xFFA5B4FC),
                    fontSize: 11,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
