import 'dart:typed_data';
import 'disease_info.dart';

class DiagnosisResult {
  final bool isRecognized;
  final String? predictedDiseaseId;
  final String predictedDisplayName;
  final double confidence;
  final double entropy;
  final Map<String, double> classProbabilities;
  final String? rejectionReason;
  final String originalImagePath;
  final Uint8List? gradCamOverlayBytes;
  final String? acneSeverity;
  final String? detectedSkinTone;
  final double? itaAngle;
  final String? clinicalSubtype;
  final DateTime timestamp;

  const DiagnosisResult({
    required this.isRecognized,
    this.predictedDiseaseId,
    required this.predictedDisplayName,
    required this.confidence,
    required this.entropy,
    required this.classProbabilities,
    this.rejectionReason,
    required this.originalImagePath,
    this.gradCamOverlayBytes,
    this.acneSeverity,
    this.detectedSkinTone,
    this.itaAngle,
    this.clinicalSubtype,
    required this.timestamp,
  });

  DiseaseInfo? get diseaseInfo =>
      predictedDiseaseId != null ? DiseaseInfo.getById(predictedDiseaseId!) : null;

  String get formattedConfidence => '${(confidence * 100).toStringAsFixed(1)}%';
  String get formattedEntropy => '${entropy.toStringAsFixed(2)} bits';
  String get formattedItaAngle => itaAngle != null ? '${itaAngle!.toStringAsFixed(1)}°' : '--';

  factory DiagnosisResult.rejection({
    required String reason,
    required String originalImagePath,
    double confidence = 0.0,
    double entropy = 0.0,
    Map<String, double> classProbabilities = const {},
    String? detectedSkinTone,
    double? itaAngle,
    Uint8List? gradCamOverlayBytes,
    String? predictedDiseaseId,
  }) {
    return DiagnosisResult(
      isRecognized: false,
      predictedDiseaseId: predictedDiseaseId,
      predictedDisplayName: 'Non-Target / Uncertain',
      confidence: confidence,
      entropy: entropy,
      classProbabilities: classProbabilities,
      rejectionReason: reason,
      originalImagePath: originalImagePath,
      detectedSkinTone: detectedSkinTone,
      itaAngle: itaAngle,
      gradCamOverlayBytes: gradCamOverlayBytes,
      timestamp: DateTime.now(),
    );
  }
}
