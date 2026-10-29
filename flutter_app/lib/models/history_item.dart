import 'dart:typed_data';
import 'diagnosis_result.dart';

class DiagnosisHistoryItem {
  final String id;
  final DateTime timestamp;
  final String originalImagePath;
  final String? gradCamOverlayPath;
  final bool isRecognized;
  final String? predictedDiseaseId;
  final String? predictedDisplayName;
  final double confidence;
  final double entropy;
  final Map<String, double> classProbabilities;
  final String? detectedSkinTone;
  final double? itaAngle;
  final String? acneSeverity;
  final String? clinicalSubtype;
  final String? rejectionReason;

  // User Feedback Loop
  final bool? isUserVerifiedCorrect;
  final String? userReportedActualDisease;
  final String? userFeedbackNotes;

  const DiagnosisHistoryItem({
    required this.id,
    required this.timestamp,
    required this.originalImagePath,
    this.gradCamOverlayPath,
    required this.isRecognized,
    this.predictedDiseaseId,
    this.predictedDisplayName,
    required this.confidence,
    required this.entropy,
    required this.classProbabilities,
    this.detectedSkinTone,
    this.itaAngle,
    this.acneSeverity,
    this.clinicalSubtype,
    this.rejectionReason,
    this.isUserVerifiedCorrect,
    this.userReportedActualDisease,
    this.userFeedbackNotes,
  });

  DiagnosisResult toDiagnosisResult({Uint8List? gradCamOverlayBytes}) {
    return DiagnosisResult(
      isRecognized: isRecognized,
      predictedDiseaseId: predictedDiseaseId,
      predictedDisplayName:
          predictedDisplayName ??
          (isRecognized
              ? (predictedDiseaseId ?? 'Target Condition')
              : 'Non-Target / Uncertain'),
      confidence: confidence,
      entropy: entropy,
      classProbabilities: classProbabilities,
      originalImagePath: originalImagePath,
      gradCamOverlayBytes: gradCamOverlayBytes,
      rejectionReason: rejectionReason,
      detectedSkinTone: detectedSkinTone,
      itaAngle: itaAngle,
      acneSeverity: acneSeverity,
      clinicalSubtype: clinicalSubtype,
      timestamp: timestamp,
    );
  }

  DiagnosisHistoryItem copyWithFeedback({
    bool? isUserVerifiedCorrect,
    String? userReportedActualDisease,
    String? userFeedbackNotes,
  }) {
    return DiagnosisHistoryItem(
      id: id,
      timestamp: timestamp,
      originalImagePath: originalImagePath,
      gradCamOverlayPath: gradCamOverlayPath,
      isRecognized: isRecognized,
      predictedDiseaseId: predictedDiseaseId,
      predictedDisplayName: predictedDisplayName,
      confidence: confidence,
      entropy: entropy,
      classProbabilities: classProbabilities,
      detectedSkinTone: detectedSkinTone,
      itaAngle: itaAngle,
      acneSeverity: acneSeverity,
      clinicalSubtype: clinicalSubtype,
      rejectionReason: rejectionReason,
      isUserVerifiedCorrect:
          isUserVerifiedCorrect ?? this.isUserVerifiedCorrect,
      userReportedActualDisease:
          userReportedActualDisease ?? this.userReportedActualDisease,
      userFeedbackNotes: userFeedbackNotes ?? this.userFeedbackNotes,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'timestamp': timestamp.toIso8601String(),
      'originalImagePath': originalImagePath,
      'gradCamOverlayPath': gradCamOverlayPath,
      'isRecognized': isRecognized,
      'predictedDiseaseId': predictedDiseaseId,
      'predictedDisplayName': predictedDisplayName,
      'confidence': confidence,
      'entropy': entropy,
      'classProbabilities': classProbabilities,
      'detectedSkinTone': detectedSkinTone,
      'itaAngle': itaAngle,
      'acneSeverity': acneSeverity,
      'clinicalSubtype': clinicalSubtype,
      'rejectionReason': rejectionReason,
      'isUserVerifiedCorrect': isUserVerifiedCorrect,
      'userReportedActualDisease': userReportedActualDisease,
      'userFeedbackNotes': userFeedbackNotes,
    };
  }

  factory DiagnosisHistoryItem.fromJson(Map<String, dynamic> json) {
    return DiagnosisHistoryItem(
      id: json['id'] as String,
      timestamp: DateTime.parse(json['timestamp'] as String),
      originalImagePath: json['originalImagePath'] as String,
      gradCamOverlayPath: json['gradCamOverlayPath'] as String?,
      isRecognized: json['isRecognized'] as bool? ?? true,
      predictedDiseaseId: json['predictedDiseaseId'] as String?,
      predictedDisplayName: json['predictedDisplayName'] as String?,
      confidence: (json['confidence'] as num).toDouble(),
      entropy: (json['entropy'] as num).toDouble(),
      classProbabilities:
          (json['classProbabilities'] as Map<String, dynamic>?)?.map(
            (k, v) => MapEntry(k, (v as num).toDouble()),
          ) ??
          {},
      detectedSkinTone: json['detectedSkinTone'] as String?,
      itaAngle: (json['itaAngle'] as num?)?.toDouble(),
      acneSeverity: json['acneSeverity'] as String?,
      clinicalSubtype: json['clinicalSubtype'] as String?,
      rejectionReason: json['rejectionReason'] as String?,
      isUserVerifiedCorrect: json['isUserVerifiedCorrect'] as bool?,
      userReportedActualDisease: json['userReportedActualDisease'] as String?,
      userFeedbackNotes: json['userFeedbackNotes'] as String?,
    );
  }
}
