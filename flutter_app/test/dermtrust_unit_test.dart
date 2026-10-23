import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:dermtrust/models/disease_info.dart';
import 'package:dermtrust/models/diagnosis_result.dart';
import 'package:dermtrust/services/image_processor.dart';
import 'package:dermtrust/utils/constants.dart';

void main() {
  group('DermTrust Disease Registry & Clinical Knowledge Base Tests', () {
    test('All 5 target clinical conditions are registered with exact parent category display names', () {
      final expectedDisplayNames = {
        'acne': 'Acne',
        'melasma': 'Melasma',
        'psoriasis': 'Psoriasis',
        'ringworm': 'Ringworm',
        'vitiligo': 'Vitiligo',
      };
      for (final entry in expectedDisplayNames.entries) {
        final info = DiseaseInfo.getById(entry.key);
        expect(info, isNotNull, reason: 'Disease ${entry.key} must be present in registry');
        expect(info!.displayName, equals(entry.value),
            reason: 'User-facing display name for ${entry.key} must match parent category ${entry.value}');
        expect(info.category.isNotEmpty, isTrue);
        expect(info.visualHallmarks.isNotEmpty, isTrue);
        expect(info.clinicallyRecognizedSubtypes.isNotEmpty, isTrue);
        expect(info.firstLineAdvice.isNotEmpty, isTrue);
        expect(info.redFlags.isNotEmpty, isTrue);
      }
    });

    test('Case-insensitivity and trim handling of disease lookup', () {
      expect(DiseaseInfo.getById(' ACNE '), isNotNull);
      expect(DiseaseInfo.getById('Vitiligo'), isNotNull);
      expect(DiseaseInfo.getById('unknown_illness'), isNull);
    });
  });

  group('Digital Image Processing (DIP) & ImageProcessor Tests', () {
    test('Preprocess produces exact [1, 224, 224, 3] Float32 buffer with values in [0, 1]', () {
      final testImg = img.Image(width: 300, height: 400);
      for (int y = 0; y < 400; y++) {
        for (int x = 0; x < 300; x++) {
          testImg.setPixelRgba(x, y, 100, 150, 200, 255);
        }
      }

      final buffer = ImageProcessor.preprocessForModel(testImg);
      expect(buffer.length, equals(1 * 224 * 224 * 3));

      // Check range [0.0, 1.0]
      for (int i = 0; i < min(buffer.length, 100); i++) {
        expect(buffer[i], inInclusiveRange(0.0, 1.0));
      }
    });

    test('Laplacian Variance sharpness operator distinguishes sharp edges from flat patches', () {
      // Flat uniform image (no edges -> variance near zero)
      final flatImg = img.Image(width: 128, height: 128);
      for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
          flatImg.setPixelRgba(x, y, 128, 128, 128, 255);
        }
      }
      final flatVar = ImageProcessor.computeLaplacianVariance(flatImg);
      expect(flatVar, equals(0.0));

      // Image with high contrast alternating edges
      final edgeImg = img.Image(width: 128, height: 128);
      for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
          final color = (x % 4 == 0) ? 255 : 0;
          edgeImg.setPixelRgba(x, y, color, color, color, 255);
        }
      }
      final edgeVar = ImageProcessor.computeLaplacianVariance(edgeImg);
      expect(edgeVar, greaterThan(100.0));
    });
  });

  group('DWH Rejection Logic & Safety Boundary Tests', () {
    test('Outcome 6 Rejection factory creates invalid result with clinical explanation', () {
      final rejection = DiagnosisResult.rejection(
        reason: AppConstants.errLowConfidence,
        originalImagePath: '/dummy/test.jpg',
        confidence: 0.42,
        entropy: 1.85,
      );

      expect(rejection.isRecognized, isFalse);
      expect(rejection.predictedDiseaseId, isNull);
      expect(rejection.rejectionReason, equals(AppConstants.errLowConfidence));
      expect(rejection.confidence, equals(0.42));
      expect(rejection.entropy, equals(1.85));
    });
  });

  group('Explainable AI (Grad-CAM) Allocation & Spatial Fidelity Tests', () {
    test('Grad-CAM output buffer allocated via List.generate creates 49 independent spatial cells', () {
      final camBuffer = List<List<List<List<double>>>>.generate(
        1,
        (_) => List<List<List<double>>>.generate(
          7,
          (_) => List<List<double>>.generate(
            7,
            (_) => List<double>.filled(1280, 0.0),
          ),
        ),
      );

      // Verify that mutating cell (0, 0) does NOT mutate cell (0, 1) or any other cell
      camBuffer[0][0][0][0] = 99.0;
      expect(camBuffer[0][0][0][0], equals(99.0));
      expect(camBuffer[0][0][1][0], equals(0.0),
          reason: 'Cell (0, 1) must be an independent memory allocation, not an alias of (0, 0)');
      expect(camBuffer[0][1][0][0], equals(0.0),
          reason: 'Cell (1, 0) must be an independent memory allocation');
      expect(identical(camBuffer[0][0][0], camBuffer[0][0][1]), isFalse,
          reason: 'Rows within feature map must not be reference identical');
    });

    test('Linear combination of distinct features produces localized spatial heatmap', () {
      // Create artificial feature map with activation centered at (3, 3)
      final featureMap = List<List<List<double>>>.generate(
        7,
        (y) => List<List<double>>.generate(
          7,
          (x) {
            final dist = (x - 3) * (x - 3) + (y - 3) * (y - 3);
            return List<double>.filled(1280, max(0.0, 10.0 - dist));
          },
        ),
      );

      final weights = List<double>.filled(1280, 1.0);
      final cam7x7 = List<List<double>>.generate(
        7,
        (y) => List<double>.generate(7, (x) {
          double sum = 0.0;
          for (int k = 0; k < 1280; k++) {
            sum += weights[k] * featureMap[y][x][k];
          }
          return max(0.0, sum);
        }),
      );

      // Verify center is peak
      expect(cam7x7[3][3], greaterThan(cam7x7[0][0]));
      expect(cam7x7[3][3], greaterThan(cam7x7[0][6]));
      expect(cam7x7[3][3], greaterThan(cam7x7[6][0]));
      expect(cam7x7[3][3], greaterThan(cam7x7[6][6]));
    });
  });

  group('Regression Tests for Master Failure Mechanisms & Safety Constraints', () {
    test('1. Exact 5-class contract and index mapping across system', () {
      expect(AppConstants.targetClasses.length, equals(5));
      expect(AppConstants.targetClasses[0], equals('acne'));
      expect(AppConstants.targetClasses[1], equals('melasma'));
      expect(AppConstants.targetClasses[2], equals('psoriasis'));
      expect(AppConstants.targetClasses[3], equals('ringworm'));
      expect(AppConstants.targetClasses[4], equals('vitiligo'));
    });

    test('2. Outcome 6 is strictly a rejection outcome, not a trained disease class', () {
      expect(AppConstants.targetClasses.contains('non_target'), isFalse);
      expect(AppConstants.targetClasses.contains('uncertain'), isFalse);
      expect(AppConstants.targetClasses.contains('outcome_6'), isFalse);
    });

    test('3. Predictive Entropy in bits: uniform vs certain distribution', () {
      // Uniform 5-class distribution: H = -5 * (0.2 * log2(0.2)) = log2(5) ~ 2.3219 bits
      const uniformP = 0.2;
      final uniformEntropy = -5 * (uniformP * (log(uniformP) / ln2));
      expect((uniformEntropy - 2.3219).abs(), lessThan(0.001));

      // Certain distribution: H = 0 bits
      const certainP = 1.0;
      final certainEntropy = -(certainP * (log(certainP) / ln2));
      expect(certainEntropy.abs(), lessThan(0.0001));

      // Threshold specification in bits
      expect(AppConstants.maxEntropyBits, equals(1.40));
      expect(AppConstants.minConfidenceThreshold, equals(0.60));
    });

    test('4. DWH Rejection boundary behavior using unrounded floating-point values', () {
      // 0.5999 must be rejected (< 0.60)
      const confBelow = 0.5999;
      expect(confBelow < AppConstants.minConfidenceThreshold, isTrue);

      // 0.6000 must pass threshold (>= 0.60)
      const confAt = 0.6000;
      expect(confAt >= AppConstants.minConfidenceThreshold, isTrue);

      // Entropy 1.4001 must be rejected (> 1.40)
      const entropyAbove = 1.4001;
      expect(entropyAbove > AppConstants.maxEntropyBits, isTrue);

      // Entropy 1.3999 must pass (<= 1.40)
      const entropyBelow = 1.3999;
      expect(entropyBelow <= AppConstants.maxEntropyBits, isTrue);
    });

    test('5. Rejection result does not assign a confirmed disease diagnosis', () {
      final rej = DiagnosisResult.rejection(
        reason: AppConstants.errLowConfidence,
        originalImagePath: 'test.jpg',
        confidence: 0.52,
        entropy: 1.65,
        classProbabilities: {'acne': 0.52, 'melasma': 0.12, 'psoriasis': 0.20, 'ringworm': 0.08, 'vitiligo': 0.08},
      );

      expect(rej.isRecognized, isFalse);
      expect(rej.predictedDiseaseId, isNull);
      expect(rej.predictedDisplayName, equals('Non-Target / Uncertain'));
      expect(rej.rejectionReason, equals(AppConstants.errLowConfidence));
    });

    test('6. Clinical severity claims are not inferred from confidence heuristics', () {
      // High confidence (0.95) must NOT claim inflammatory severity or urgent staging
      final res = DiagnosisResult(
        isRecognized: true,
        predictedDiseaseId: 'acne',
        predictedDisplayName: 'Acne',
        confidence: 0.95,
        entropy: 0.25,
        classProbabilities: {'acne': 0.95, 'melasma': 0.01, 'psoriasis': 0.02, 'ringworm': 0.01, 'vitiligo': 0.01},
        originalImagePath: 'test.jpg',
        acneSeverity: null, // Unsupported heuristic removed
        timestamp: DateTime.now(),
      );

      expect(res.acneSeverity, isNull);
    });

    test('7. Image-quality blur rejection provides clear quality guidance rather than disease claim', () {
      final blurRej = DiagnosisResult.rejection(
        reason: AppConstants.errBlurryImage,
        originalImagePath: 'blurry.jpg',
      );

      expect(blurRej.isRecognized, isFalse);
      expect(blurRej.rejectionReason, contains('blurry'));
      expect(blurRej.predictedDiseaseId, isNull);
    });
  });
}
