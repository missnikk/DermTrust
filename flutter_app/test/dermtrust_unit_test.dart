import 'dart:math';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:dermtrust/models/disease_info.dart';
import 'package:dermtrust/models/diagnosis_result.dart';
import 'package:dermtrust/services/image_processor.dart';
import 'package:dermtrust/utils/constants.dart';

void main() {
  group('DermTrust Disease Registry & Clinical Knowledge Base Tests', () {
    test('All 5 target clinical conditions are registered', () {
      final requiredDiseases = ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo'];
      for (final id in requiredDiseases) {
        final info = DiseaseInfo.getById(id);
        expect(info, isNotNull, reason: 'Disease $id must be present in registry');
        expect(info!.displayName.isNotEmpty, isTrue);
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
}
