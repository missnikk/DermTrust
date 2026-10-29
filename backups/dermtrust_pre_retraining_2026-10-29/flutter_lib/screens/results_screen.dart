import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/diagnosis_result.dart';
import '../models/disease_info.dart';
import '../services/history_service.dart';
import '../utils/app_theme.dart';
import '../widgets/confidence_indicator.dart';
import '../widgets/disease_card.dart';
import '../widgets/gradcam_viewer.dart';
import '../widgets/safety_banner.dart';

class ResultsScreen extends StatefulWidget {
  final DiagnosisResult result;
  final String? historyId;

  const ResultsScreen({super.key, required this.result, this.historyId});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  String? _currentHistoryId;
  bool? _isUserVerifiedCorrect;
  String? _userActualDisease;
  Timer? _feedbackTimer;
  bool _feedbackDialogShown = false;

  @override
  void initState() {
    super.initState();
    _currentHistoryId = widget.historyId;

    // Automatically record to History if new diagnosis
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      if (_currentHistoryId == null) {
        final historyService = Provider.of<HistoryService>(context, listen: false);
        final item = await historyService.addRecord(widget.result);
        if (mounted) {
          setState(() {
            _currentHistoryId = item.id;
          });
        }
      }

      // Schedule popup after diagnosis review (5 seconds)
      _feedbackTimer = Timer(const Duration(seconds: 4), () {
        if (mounted && !_feedbackDialogShown && _isUserVerifiedCorrect == null) {
          _feedbackDialogShown = true;
          _showFeedbackModal(context);
        }
      });
    });
  }

  @override
  void dispose() {
    _feedbackTimer?.cancel();
    super.dispose();
  }

  Future<void> _submitFeedback(bool isCorrect, {String? actualCondition}) async {
    setState(() {
      _isUserVerifiedCorrect = isCorrect;
      _userActualDisease = actualCondition;
    });

    if (_currentHistoryId != null) {
      final historyService = Provider.of<HistoryService>(context, listen: false);
      await historyService.updateFeedback(
        historyId: _currentHistoryId!,
        isCorrect: isCorrect,
        userReportedActualDisease: actualCondition,
      );
    }

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            isCorrect
              ? 'Thank you! Accuracy confirmation logged.'
              : 'Clinical discrepancy logged (Actual: ${actualCondition ?? 'Other'}). This will be used to calibrate model weights.',
          ),
          backgroundColor: isCorrect ? AppTheme.successGreen : AppTheme.accentCyan,
          duration: const Duration(seconds: 3),
        ),
      );
    }
  }

  void _showFeedbackModal(BuildContext context) {
    String selectedTrueCondition = 'Melasma';
    final conditions = ['Melasma', 'Vitiligo', 'Acne', 'Psoriasis', 'Ringworm', 'Non-Target / Other'];

    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.surfaceElevated,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: AppTheme.surfaceBorder,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Row(
                    children: [
                      Icon(Icons.rate_review_outlined, color: AppTheme.accentCyan, size: 22),
                      SizedBox(width: 10),
                      Text(
                        'Clinical Accuracy Verification',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 17,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Was this lesion correctly diagnosed as ${widget.result.isRecognized ? widget.result.predictedDisplayName : 'Not Recognised'}?',
                    style: const TextStyle(color: AppTheme.textSecondary, fontSize: 13, height: 1.4),
                  ),
                  const SizedBox(height: 20),

                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: AppTheme.successGreen),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          onPressed: () {
                            Navigator.of(ctx).pop();
                            _submitFeedback(true);
                          },
                          icon: const Icon(Icons.check_circle_rounded, color: AppTheme.successGreen, size: 18),
                          label: const Text('Yes, Correct', style: TextStyle(color: AppTheme.successGreen, fontWeight: FontWeight.bold)),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: AppTheme.dangerRose),
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                          onPressed: () {
                            setModalState(() {
                              _feedbackDialogShown = true;
                            });
                            // Show condition selection
                            showDialog(
                              context: context,
                              builder: (dCtx) => AlertDialog(
                                backgroundColor: AppTheme.surfaceElevated,
                                title: const Text('What is the actual condition?', style: TextStyle(color: AppTheme.textPrimary, fontSize: 16)),
                                content: DropdownButtonFormField<String>(
                                  dropdownColor: AppTheme.surfaceElevated,
                                  value: selectedTrueCondition,
                                  items: conditions
                                      .map((c) => DropdownMenuItem(value: c, child: Text(c, style: const TextStyle(color: AppTheme.textPrimary))))
                                      .toList(),
                                  onChanged: (val) {
                                    if (val != null) selectedTrueCondition = val;
                                  },
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.of(dCtx).pop(),
                                    child: const Text('Cancel', style: TextStyle(color: AppTheme.textSecondary)),
                                  ),
                                  ElevatedButton(
                                    style: ElevatedButton.styleFrom(backgroundColor: AppTheme.primaryBlue),
                                    onPressed: () {
                                      Navigator.of(dCtx).pop();
                                      Navigator.of(ctx).pop();
                                      _submitFeedback(false, actualCondition: selectedTrueCondition);
                                    },
                                    child: const Text('Submit Feedback', style: TextStyle(color: Colors.white)),
                                  ),
                                ],
                              ),
                            );
                          },
                          icon: const Icon(Icons.cancel_rounded, color: AppTheme.dangerRose, size: 18),
                          label: const Text('No, Incorrect', style: TextStyle(color: AppTheme.dangerRose, fontWeight: FontWeight.bold)),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final result = widget.result;

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
            // User Feedback Interactive Banner Card
            _buildInteractiveFeedbackCard(),

            const SizedBox(height: 14),

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
                  detectedSkinTone: result.detectedSkinTone,
                  itaAngle: result.itaAngle,
                  clinicalSubtype: result.clinicalSubtype,
                ),

              const SizedBox(height: 16),

              // AI Clinical Differential Diagnosis Card (Melasma/Acne/Vitiligo/Psoriasis)
              _buildClinicalDifferentialCard(result),

              const SizedBox(height: 16),

              // Probability Breakdown Across Target Classes
              _buildProbabilityBreakdown(result.classProbabilities),
            ] else ...[
              // Rejection / Non-Target Condition Path (Outcome 6)
              _buildRejectionBanner(context),

              const SizedBox(height: 16),

              // Grad-CAM Visual Heatmap (or image fallback)
              if (result.gradCamOverlayBytes != null)
                GradCamViewer(
                  originalImagePath: result.originalImagePath,
                  gradCamBytes: result.gradCamOverlayBytes,
                  diseaseName: result.predictedDiseaseId != null
                      ? (DiseaseInfo.getById(result.predictedDiseaseId!)?.displayName ?? result.predictedDisplayName)
                      : result.predictedDisplayName,
                )
              else
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

  Widget _buildInteractiveFeedbackCard() {
    if (_isUserVerifiedCorrect != null) {
      final isCorrect = _isUserVerifiedCorrect!;
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: isCorrect ? AppTheme.successGreen.withOpacity(0.12) : AppTheme.accentCyan.withOpacity(0.12),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: isCorrect ? AppTheme.successGreen.withOpacity(0.4) : AppTheme.accentCyan.withOpacity(0.4)),
        ),
        child: Row(
          children: [
            Icon(
              isCorrect ? Icons.verified_rounded : Icons.info_outline_rounded,
              color: isCorrect ? AppTheme.successGreen : AppTheme.accentCyan,
              size: 20,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                isCorrect
                    ? 'Clinical Accuracy Verified (Logged in History)'
                    : 'Reported True Condition: ${_userActualDisease ?? 'Other'} (Logged)',
                style: TextStyle(
                  color: isCorrect ? AppTheme.successGreen : AppTheme.accentCyan,
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.surfaceBorder),
      ),
      child: Row(
        children: [
          const Icon(Icons.help_outline_rounded, color: AppTheme.accentCyan, size: 20),
          const SizedBox(width: 10),
          const Expanded(
            child: Text(
              'Was this diagnosis accurate?',
              style: TextStyle(color: AppTheme.textPrimary, fontSize: 13, fontWeight: FontWeight.w500),
            ),
          ),
          TextButton(
            onPressed: () => _submitFeedback(true),
            style: TextButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 10)),
            child: const Text('Yes', style: TextStyle(color: AppTheme.successGreen, fontWeight: FontWeight.bold)),
          ),
          TextButton(
            onPressed: () => _showFeedbackModal(context),
            style: TextButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 10)),
            child: const Text('No', style: TextStyle(color: AppTheme.dangerRose, fontWeight: FontWeight.bold)),
          ),
        ],
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
            widget.result.rejectionReason ??
                'The system declined to force a classification because the lesion does not match the 5 supported categories with reliable statistical confidence.',
            style: const TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 13,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 12),
          if (widget.result.detectedSkinTone != null) ...[
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              margin: const EdgeInsets.only(bottom: 10),
              decoration: BoxDecoration(
                color: AppTheme.surfaceElevated,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppTheme.surfaceBorder),
              ),
              child: Row(
                children: [
                  const Icon(Icons.palette_outlined, color: AppTheme.accentCyan, size: 14),
                  const SizedBox(width: 8),
                  Text(
                    'Assessed Skin Tone: ${widget.result.detectedSkinTone} (${widget.result.formattedItaAngle})',
                    style: const TextStyle(
                      color: AppTheme.textSecondary,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ],
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

  Widget _buildClinicalDifferentialCard(DiagnosisResult result) {
    if (result.classProbabilities.isEmpty) return const SizedBox.shrink();

    final sorted = result.classProbabilities.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    if (sorted.length < 2) return const SizedBox.shrink();

    final top1 = sorted[0].key.toLowerCase();
    final top2 = sorted[1].key.toLowerCase();
    final p1 = sorted[0].value;
    final p2 = sorted[1].value;

    String? differentialTitle;
    String? differentialInsight;
    String? distinctionRule;
    Color badgeColor = AppTheme.primaryBlue;

    // 1. Melasma vs Acne Differential
    if ((top1.contains('acne') && top2.contains('melasma')) ||
        (top1.contains('melasma') && top2.contains('acne')) ||
        (top1.contains('melasma') && p1 < 0.85) ||
        (top1.contains('acne') && (result.classProbabilities['melasma'] ?? 0.0) > 0.12)) {
      differentialTitle = 'Differential: Melasma vs. Acne Hyperpigmentation';
      differentialInsight =
          'Facial pigmentation often shares overlapping visual fields. Melasma presents as flat, non-inflammatory brownish patches (malar/centrofacial), whereas Acne presents with palpable follicular papules, pustules, or post-inflammatory erythema.';
      distinctionRule =
          'Look for lesion elevation: If the skin texture is completely flat and smooth with reticular pigment, Melasma is the true clinical diagnosis.';
      badgeColor = const Color(0xFFD97706); // Amber
    }
    // 2. Vitiligo vs Psoriasis Differential
    else if ((top1.contains('psoriasis') && top2.contains('vitiligo')) ||
             (top1.contains('vitiligo') && top2.contains('psoriasis')) ||
             (top1.contains('vitiligo') && p1 < 0.85) ||
             (top1.contains('psoriasis') && (result.classProbabilities['vitiligo'] ?? 0.0) > 0.12)) {
      differentialTitle = 'Differential: Vitiligo vs. Psoriasis';
      differentialInsight =
          'High luminance lesions can reflect light similarly. Vitiligo is marked by complete epidermal melanin loss (porcelain-white macules with smooth texture), whereas Psoriasis presents with thick, elevated erythematous plaques covered in silvery micaceous scales.';
      distinctionRule =
          'Look for surface scaling: If there is zero flaking or scaling and the skin is depigmented chalk-white, Vitiligo is the true clinical diagnosis.';
      badgeColor = const Color(0xFF0284C7); // Cyan-Blue
    }
    // 3. Ringworm vs Psoriasis Differential
    else if ((top1.contains('ringworm') && top2.contains('psoriasis')) ||
             (top1.contains('psoriasis') && top2.contains('ringworm'))) {
      differentialTitle = 'Differential: Ringworm (Tinea) vs. Plaque Psoriasis';
      differentialInsight =
          'Both conditions feature scaly erythematous margins. Tinea corporis shows classic annular configuration with an active centrifugal advancing border and central clearing.';
      distinctionRule =
          'Look for central clearing: An expanding ring with clear central skin confirms Ringworm; uniform plaques confirm Psoriasis.';
      badgeColor = const Color(0xFF10B981); // Emerald
    }

    if (differentialTitle == null) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surfaceElevated,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: badgeColor.withOpacity(0.5), width: 1.2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.compare_arrows_rounded, color: badgeColor, size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  differentialTitle,
                  style: TextStyle(
                    color: badgeColor,
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: badgeColor.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text(
                  'CDS Rule',
                  style: TextStyle(fontSize: 10, fontWeight: FontWeight.bold, color: Colors.white70),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            differentialInsight!,
            style: const TextStyle(color: AppTheme.textSecondary, fontSize: 12, height: 1.4),
          ),
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: AppTheme.background,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppTheme.surfaceBorder),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.lightbulb_outline_rounded, color: AppTheme.accentCyan, size: 16),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    distinctionRule!,
                    style: const TextStyle(
                      color: AppTheme.textPrimary,
                      fontSize: 11.5,
                      fontWeight: FontWeight.w500,
                      height: 1.35,
                    ),
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
            'Confidence distribution across classes (including non-target detection).',
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
                        e.value >= 0.40 ? AppTheme.primaryBlue : AppTheme.surfaceBorder,
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
