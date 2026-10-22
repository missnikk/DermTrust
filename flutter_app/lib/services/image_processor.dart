import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import '../utils/constants.dart';

class ImageValidationResult {
  final bool isValid;
  final String? errorMessage;
  final img.Image? decodedImage;
  final double blurScore;

  const ImageValidationResult({
    required this.isValid,
    this.errorMessage,
    this.decodedImage,
    required this.blurScore,
  });
}

class ImageProcessor {
  /// Decodes and validates the input image file for resolution, format, and blur.
  static Future<ImageValidationResult> validateImage(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);

      if (image == null) {
        return const ImageValidationResult(
          isValid: false,
          errorMessage: 'Failed to decode image file. Please provide a standard JPEG or PNG photo.',
          blurScore: 0.0,
        );
      }

      if (image.width < 100 || image.height < 100) {
        return const ImageValidationResult(
          isValid: false,
          errorMessage: 'Image resolution is too low (<100px) to distinguish clinical lesion morphology.',
          blurScore: 0.0,
        );
      }

      // Check Blur using Laplacian Variance on downscaled grayscale
      final blurScore = computeLaplacianVariance(image);
      if (blurScore < AppConstants.minBlurLaplacianVariance) {
        return ImageValidationResult(
          isValid: false,
          errorMessage: 'Image appears excessively blurry (Laplacian variance: ${blurScore.toStringAsFixed(1)} < ${AppConstants.minBlurLaplacianVariance.toStringAsFixed(0)}). Please capture a well-focused image.',
          decodedImage: image,
          blurScore: blurScore,
        );
      }

      return ImageValidationResult(
        isValid: true,
        decodedImage: image,
        blurScore: blurScore,
      );
    } catch (e) {
      return ImageValidationResult(
        isValid: false,
        errorMessage: 'Error processing image: $e',
        blurScore: 0.0,
      );
    }
  }

  /// Calculates variance of discrete 2D Laplacian operator as a sharpness proxy.
  static double computeLaplacianVariance(img.Image src) {
    // Resize to smaller thumbnail for fast O(N) mobile computation
    final thumbnail = img.copyResize(src, width: 128, height: 128);
    final grayscale = img.grayscale(thumbnail);

    final w = grayscale.width;
    final h = grayscale.height;
    final values = <double>[];
    double sum = 0.0;

    for (int y = 1; y < h - 1; y++) {
      for (int x = 1; x < w - 1; x++) {
        // Standard discrete 3x3 Laplacian kernel
        //  0  1  0
        //  1 -4  1
        //  0  1  0
        final center = grayscale.getPixel(x, y).r;
        final up = grayscale.getPixel(x, y - 1).r;
        final down = grayscale.getPixel(x, y + 1).r;
        final left = grayscale.getPixel(x - 1, y).r;
        final right = grayscale.getPixel(x + 1, y).r;

        final lap = (up + down + left + right) - (4 * center);
        final lapD = lap.toDouble();
        values.add(lapD);
        sum += lapD;
      }
    }

    if (values.isEmpty) return 0.0;
    final mean = sum / values.length;
    double varSum = 0.0;
    for (final v in values) {
      varSum += (v - mean) * (v - mean);
    }
    return varSum / values.length;
  }

  /// Converts image to [1, 224, 224, 3] Float32List normalized to [0.0, 1.0].
  static Float32List preprocessForModel(img.Image src) {
    final resized = img.copyResize(src, width: 224, height: 224);
    final buffer = Float32List(1 * 224 * 224 * 3);
    int bufferIdx = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        // RGB order, normalized to [0.0, 1.0]
        buffer[bufferIdx++] = pixel.r / 255.0;
        buffer[bufferIdx++] = pixel.g / 255.0;
        buffer[bufferIdx++] = pixel.b / 255.0;
      }
    }
    return buffer;
  }

  /// Generates a Jet-style heatmap overlay on the original 224x224 image.
  static Uint8List createGradCamOverlay({
    required img.Image original,
    required List<List<double>> heatmap7x7,
    double alpha = 0.35,
  }) {
    final base = img.copyResize(original, width: 224, height: 224);
    final overlayImg = img.Image(width: 224, height: 224);

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        // Bilinear interpolation from 7x7 grid
        final gx = (x / 224.0) * 6.0;
        final gy = (y / 224.0) * 6.0;
        final x0 = gx.floor().clamp(0, 5);
        final y0 = gy.floor().clamp(0, 5);
        final x1 = x0 + 1;
        final y1 = y0 + 1;

        final fx = gx - x0;
        final fy = gy - y0;

        final v00 = heatmap7x7[y0][x0];
        final v10 = heatmap7x7[y0][x1];
        final v01 = heatmap7x7[y1][x0];
        final v11 = heatmap7x7[y1][x1];

        final val = (v00 * (1 - fx) * (1 - fy)) +
            (v10 * fx * (1 - fy)) +
            (v01 * (1 - fx) * fy) +
            (v11 * fx * fy);

        final clamped = val.clamp(0.0, 1.0);

        // Jet colormap RGB approximation
        // Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
        final jetColor = getJetColor(clamped);
        final origPixel = base.getPixel(x, y);

        final r = ((1 - alpha) * origPixel.r + alpha * jetColor[0]).round().clamp(0, 255);
        final g = ((1 - alpha) * origPixel.g + alpha * jetColor[1]).round().clamp(0, 255);
        final b = ((1 - alpha) * origPixel.b + alpha * jetColor[2]).round().clamp(0, 255);

        overlayImg.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return Uint8List.fromList(img.encodePng(overlayImg));
  }

  static List<int> getJetColor(double v) {
    double r = 0.0, g = 0.0, b = 0.0;
    if (v < 0.125) {
      r = 0.0;
      g = 0.0;
      b = 0.5 + (v / 0.125) * 0.5;
    } else if (v < 0.375) {
      r = 0.0;
      g = (v - 0.125) / 0.25;
      b = 1.0;
    } else if (v < 0.625) {
      r = (v - 0.375) / 0.25;
      g = 1.0;
      b = 1.0 - (v - 0.375) / 0.25;
    } else if (v < 0.875) {
      r = 1.0;
      g = 1.0 - (v - 0.625) / 0.25;
      b = 0.0;
    } else {
      r = 1.0 - 0.5 * ((v - 0.875) / 0.125);
      g = 0.0;
      b = 0.0;
    }
    return [
      (r * 255).round().clamp(0, 255),
      (g * 255).round().clamp(0, 255),
      (b * 255).round().clamp(0, 255),
    ];
  }
}
