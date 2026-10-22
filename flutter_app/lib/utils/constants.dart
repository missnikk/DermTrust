import 'package:flutter/material.dart';

class AppConstants {
  static const String appName = 'DermTrust';
  static const String appTagline = 'Fairness-Evaluated, Explainable AI Skin Decision Support';
  static const String appVersion = '1.0.0 (Final-Year Major Project)';

  // Assets
  static const String classifierModelAsset = 'assets/models/dermtrust_classifier.tflite';
  static const String camModelAsset = 'assets/models/dermtrust_cam.tflite';
  static const String camWeightsAsset = 'assets/models/cam_weights.json';
  static const String metadataAsset = 'assets/models/model_metadata.json';

  // Input Image Specification
  static const int inputSize = 224;
  static const int numChannels = 3;

  // Rejection & Uncertainty Thresholds (DWH Tier 2)
  static const double minConfidenceThreshold = 0.60; // 60%
  static const double maxEntropyBits = 1.40;          // Shannon entropy
  static const double minBlurLaplacianVariance = 30.0;

  // Supported Disease Classes (The 5 Target Clinical Classes)
  static const List<String> targetClasses = [
    'acne',
    'melasma',
    'psoriasis',
    'ringworm',
    'vitiligo',
  ];

  // Rejection Reason Messages
  static const String errLowConfidence =
      'Prediction confidence is below the clinical decision support threshold (60%). The lesion may represent a non-target condition outside the supported disease categories.';
  static const String errHighEntropy =
      'High model uncertainty detected across multiple conditions. A definitive prediction cannot be reliably established.';
  static const String errBlurryImage =
      'The input image appears blurry or lacks adequate optical focus. Please retake under stable lighting.';
  static const String errNonSkin =
      'The image does not appear to contain a sufficiently clear, illuminated skin lesion region.';

  // Mandatory Clinical Disclaimer
  static const String medicalDisclaimer =
      'DermTrust is an academic research prototype and clinical decision-support tool. It does not provide definitive medical diagnoses, prescribe treatments, or replace an evaluation by a board-certified dermatologist. Always consult a licensed medical professional for clinical diagnosis and care.';
}
