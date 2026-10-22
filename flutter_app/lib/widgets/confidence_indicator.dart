import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class ConfidenceIndicator extends StatelessWidget {
  final double confidence;
  final double entropy;
  final bool isRecognized;

  const ConfidenceIndicator({
    super.key,
    required this.confidence,
    required this.entropy,
    required this.isRecognized,
  });

  Color _getColor() {
    if (!isRecognized) return AppTheme.dangerRose;
    if (confidence >= 0.80) return AppTheme.successGreen;
    if (confidence >= 0.60) return AppTheme.warningAmber;
    return AppTheme.dangerRose;
  }

  String _getStatusLabel() {
    if (!isRecognized) return 'UNRELIABLE / REJECTED';
    if (confidence >= 0.80) return 'HIGH CONFIDENCE';
    if (confidence >= 0.60) return 'MODERATE CONFIDENCE';
    return 'LOW CONFIDENCE';
  }

  @override
  Widget build(BuildContext context) {
    final color = _getColor();
    final pct = (confidence * 100).toStringAsFixed(1);

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'MODEL CONFIDENCE',
                style: TextStyle(
                  color: AppTheme.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.8,
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                  _getStatusLabel(),
                  style: TextStyle(
                    color: color,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                '$pct%',
                style: TextStyle(
                  color: color,
                  fontSize: 26,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 12),
              Text(
                'Entropy: ${entropy.toStringAsFixed(2)} bits',
                style: const TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: 12,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: confidence.clamp(0.0, 1.0),
              minHeight: 6,
              backgroundColor: AppTheme.surfaceBorder,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ],
      ),
    );
  }
}
