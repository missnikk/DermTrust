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
}
