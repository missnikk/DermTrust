import 'package:flutter/material.dart';
import '../models/disease_info.dart';
import '../utils/app_theme.dart';

class DiseaseCard extends StatelessWidget {
  final DiseaseInfo info;
  final String? acneSeverity;
  final String? detectedSkinTone;
  final double? itaAngle;
  final String? clinicalSubtype;

  const DiseaseCard({
    super.key,
    required this.info,
    this.acneSeverity,
    this.detectedSkinTone,
    this.itaAngle,
    this.clinicalSubtype,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Category Header
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppTheme.surfaceElevated,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppTheme.surfaceBorder),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                info.category.toUpperCase(),
                style: const TextStyle(
                  color: AppTheme.accentCyan,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.8,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                info.displayName,
                style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (detectedSkinTone != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppTheme.accentCyan.withOpacity(0.12),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: AppTheme.accentCyan.withOpacity(0.4)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.palette_outlined, color: AppTheme.accentCyan, size: 14),
                      const SizedBox(width: 6),
                      Text(
                        'Assessed Skin Tone: $detectedSkinTone ${itaAngle != null ? "(${itaAngle!.toStringAsFixed(1)}°)" : ""}',
                        style: const TextStyle(
                          color: AppTheme.accentCyan,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              if (clinicalSubtype != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppTheme.primaryBlue.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: AppTheme.primaryBlue.withOpacity(0.4)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.biotech_outlined, color: AppTheme.primaryBlue, size: 14),
                      const SizedBox(width: 6),
                      Flexible(
                        child: Text(
                          'Subtype: $clinicalSubtype',
                          style: const TextStyle(
                            color: AppTheme.primaryBlue,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              if (acneSeverity != null) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppTheme.warningAmber.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: AppTheme.warningAmber.withOpacity(0.4)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.assessment_outlined, color: AppTheme.warningAmber, size: 14),
                      const SizedBox(width: 6),
                      Flexible(
                        child: Text(
                          'Severity: $acneSeverity',
                          style: const TextStyle(
                            color: AppTheme.warningAmber,
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 8),
              Text(
                info.description,
                style: const TextStyle(
                  color: AppTheme.textSecondary,
                  fontSize: 13,
                  height: 1.45,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 12),

        // Visual Hallmarks
        _buildSectionCard(
          title: 'Visual Hallmarks Detected by Model',
          icon: Icons.checklist_rtl_rounded,
          iconColor: AppTheme.primaryBlue,
          content: Column(
            children: info.visualHallmarks
                .map((h) => _buildBulletPoint(h, Icons.circle, AppTheme.primaryBlue))
                .toList(),
          ),
        ),

        const SizedBox(height: 12),

        // Clinically Recognized Subtypes
        _buildSectionCard(
          title: 'Clinically Recognized Subtypes',
          icon: Icons.category_outlined,
          iconColor: AppTheme.accentCyan,
          content: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Subtypes must be clinically confirmed via in-person dermatological examination:',
                style: TextStyle(color: AppTheme.textMuted, fontSize: 11, fontStyle: FontStyle.italic),
              ),
              const SizedBox(height: 6),
              ...info.clinicallyRecognizedSubtypes
                  .map((s) => _buildBulletPoint(s, Icons.arrow_right_rounded, AppTheme.accentCyan)),
            ],
          ),
        ),

        const SizedBox(height: 12),

        // First Line Advice
        _buildSectionCard(
          title: 'Initial General Care Guidance',
          icon: Icons.healing_outlined,
          iconColor: AppTheme.successGreen,
          content: Text(
            info.firstLineAdvice,
            style: const TextStyle(color: AppTheme.textSecondary, fontSize: 13, height: 1.45),
          ),
        ),

        if (info.id == 'acne' || info.id == 'melasma') ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFF334155)),
            ),
            child: const Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.lightbulb_outline_rounded, color: AppTheme.warningAmber, size: 16),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Clinical Note: Facial hyperpigmentation (such as melasma) and post-inflammatory pigmentation from acne can share visual characteristics in clinical photography. An in-person dermatological assessment is recommended for definitive differentiation.',
                    style: TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 11,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        if (info.id == 'psoriasis') ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFF334155)),
            ),
            child: const Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.lightbulb_outline_rounded, color: AppTheme.warningAmber, size: 16),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Clinical Note: Other scaly or inflammatory conditions (such as eczema or contact dermatitis) can present visual plaque features similar to psoriasis. In-person clinical examination is necessary for differential diagnosis.',
                    style: TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 11,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        if (info.id == 'ringworm') ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFF334155)),
            ),
            child: const Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.lightbulb_outline_rounded, color: AppTheme.warningAmber, size: 16),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Clinical Note: Tinea corporis (ringworm) shares annular erythematous scaling borders with plaque psoriasis and nummular eczema. In-person direct potassium hydroxide (KOH) microscopy is clinically indicated to confirm fungal etiology.',
                    style: TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 11,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        if (info.id == 'vitiligo') ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFF1E293B),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFF334155)),
            ),
            child: const Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.lightbulb_outline_rounded, color: AppTheme.warningAmber, size: 16),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Clinical Note: Depigmented macules on lighter skin phototypes (Fitzpatrick I-II) exhibit reduced optical contrast in digital photography. Wood\'s lamp dermatoscopy and clinical evaluation are recommended to differentiate vitiligo from post-inflammatory hypopigmentation or pityriasis alba.',
                    style: TextStyle(
                      color: Color(0xFFCBD5E1),
                      fontSize: 11,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],

        const SizedBox(height: 12),

        // Red Flags
        _buildSectionCard(
          title: 'When to Seek Urgent Medical Care',
          icon: Icons.warning_amber_rounded,
          iconColor: AppTheme.dangerRose,
          content: Column(
            children: info.redFlags
                .map((r) => _buildBulletPoint(r, Icons.priority_high_rounded, AppTheme.dangerRose))
                .toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildSectionCard({
    required String title,
    required IconData icon,
    required Color iconColor,
    required Widget content,
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
              Icon(icon, color: iconColor, size: 16),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  color: AppTheme.textPrimary,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          content,
        ],
      ),
    );
  }

  Widget _buildBulletPoint(String text, IconData icon, Color color) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 4, right: 8),
            child: Icon(icon, size: 8, color: color),
          ),
          Expanded(
            child: Text(
              text,
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 12,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
