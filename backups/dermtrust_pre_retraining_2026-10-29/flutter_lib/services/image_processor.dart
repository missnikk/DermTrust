import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import '../utils/constants.dart';

class SkinToneEstimate {
  final double itaAngle;
  final String toneGroup;
  final bool isBoundary;

  const SkinToneEstimate({
    required this.itaAngle,
    required this.toneGroup,
    required this.isBoundary,
  });

  String get formattedAngle => '${itaAngle.toStringAsFixed(1)}°';
}

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

      final orientedImage = img.bakeOrientation(image);

      if (orientedImage.width < 100 || orientedImage.height < 100) {
        return const ImageValidationResult(
          isValid: false,
          errorMessage: 'Image resolution is too low (<100px) to distinguish clinical lesion morphology.',
          blurScore: 0.0,
        );
      }

      // Check Blur using Laplacian Variance on downscaled grayscale
      final blurScore = computeLaplacianVariance(orientedImage);
      if (blurScore < AppConstants.minBlurLaplacianVariance) {
        return ImageValidationResult(
          isValid: false,
          errorMessage: 'Image appears excessively blurry (Laplacian variance: ${blurScore.toStringAsFixed(1)} < ${AppConstants.minBlurLaplacianVariance.toStringAsFixed(0)}). Please capture a well-focused image.',
          decodedImage: orientedImage,
          blurScore: blurScore,
        );
      }

      return ImageValidationResult(
        isValid: true,
        decodedImage: orientedImage,
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
  /// Uses bilinear interpolation to ensure exact numerical parity with Python training.
  static Float32List preprocessForModel(img.Image src) {
    final resized = img.copyResize(
      src,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );
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

  /// Computes Individual Typology Angle (ITA) in degrees from healthy skin pixels
  /// in CIE L*a*b* space, categorizing into Fitzpatrick groups (B3, B4).
  static SkinToneEstimate estimateSkinTone(img.Image src) {
    final thumb = img.copyResize(
      src,
      width: 128,
      height: 128,
      interpolation: img.Interpolation.linear,
    );
    final w = thumb.width;
    final h = thumb.height;

    final lValues = <double>[];
    final bValues = <double>[];

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        final pixel = thumb.getPixel(x, y);
        final r = pixel.r.toDouble();
        final g = pixel.g.toDouble();
        final b = pixel.b.toDouble();

        // Biological human skin heuristic: R > G and G >= B * 0.9 and R - G >= 5
        final bool isSkinRgb = (r > g) && (g >= b * 0.9) && (r - g >= 5.0);
        final bool isIntensity = (r > 40.0) && (r < 250.0) && (b < 235.0);

        final tot = r + g + b + 1e-5;
        final rNorm = r / tot;
        final gNorm = g / tot;
        final bool isChroma = (rNorm > 0.33) && (rNorm < 0.65) && (gNorm > 0.25) && (gNorm < 0.40);

        if (isSkinRgb && isIntensity && isChroma) {
          final lab = _rgbToCieLab(r / 255.0, g / 255.0, b / 255.0);
          lValues.add(lab[0]);
          bValues.add(lab[2]);
        }
      }
    }

    // Fallback if skin mask captures too few pixels
    if (lValues.length < 50) {
      for (int y = (h * 0.25).toInt(); y < (h * 0.75).toInt(); y++) {
        for (int x = (w * 0.25).toInt(); x < (w * 0.75).toInt(); x++) {
          final pixel = thumb.getPixel(x, y);
          final lab = _rgbToCieLab(pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0);
          lValues.add(lab[0]);
          bValues.add(lab[2]);
        }
      }
    }

    lValues.sort();
    bValues.sort();

    // 75th percentile of L* to target healthy background skin
    final p75Idx = (lValues.length * 0.75).toInt().clamp(0, lValues.length - 1);
    final medBIdx = (bValues.length * 0.50).toInt().clamp(0, bValues.length - 1);

    final lStar = lValues.isNotEmpty ? lValues[p75Idx] : 50.0;
    double bStar = bValues.isNotEmpty ? bValues[medBIdx] : 15.0;
    if (bStar.abs() < 1e-4) bStar = 1e-4;

    final itaRad = atan2(lStar - 50.0, bStar);
    final itaDeg = itaRad * (180.0 / pi);

    final bool isBoundary = (itaDeg >= 21.0 && itaDeg <= 41.0);
    String toneGroup;
    if (itaDeg >= 41.0) {
      toneGroup = 'Light Skin (FST I–III)';
    } else if (itaDeg < 21.0) {
      toneGroup = 'Darker Skin (FST IV–VI)';
    } else {
      toneGroup = itaDeg >= 31.0
          ? 'Intermediate Light (FST III)'
          : 'Intermediate Dark (FST IV)';
    }

    return SkinToneEstimate(
      itaAngle: itaDeg,
      toneGroup: toneGroup,
      isBoundary: isBoundary,
    );
  }

  /// Converts single normalized RGB [0, 1] to CIE L*a*b* [L*, a*, b*].
  static List<double> _rgbToCieLab(double r, double g, double b) {
    final rLin = r > 0.04045 ? pow((r + 0.055) / 1.055, 2.4).toDouble() : r / 12.92;
    final gLin = g > 0.04045 ? pow((g + 0.055) / 1.055, 2.4).toDouble() : g / 12.92;
    final bLin = b > 0.04045 ? pow((b + 0.055) / 1.055, 2.4).toDouble() : b / 12.92;

    final x = (rLin * 0.4124564 + gLin * 0.3575761 + bLin * 0.1804375) / 0.95047;
    final y = (rLin * 0.2126729 + gLin * 0.7151522 + bLin * 0.0721750) / 1.00000;
    final z = (rLin * 0.0193339 + gLin * 0.1191920 + bLin * 0.9503041) / 1.08883;

    const delta = 6.0 / 29.0;
    const delta3 = delta * delta * delta;
    double f(double t) => t > delta3 ? pow(t, 1.0 / 3.0).toDouble() : (t / (3.0 * delta * delta)) + (4.0 / 29.0);

    final fx = f(x);
    final fy = f(y);
    final fz = f(z);

    final lStar = 116.0 * fy - 16.0;
    final aStar = 500.0 * (fx - fy);
    final bStar = 200.0 * (fy - fz);

    return [lStar, aStar, bStar];
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

        // Active Clinical Thermal Colormap (Zero Blue Tint)
        // Background (low activation): zero tint so healthy skin is 100% natural and visible
        // Mid-low (0.20 - 0.45): vibrant lime green -> chartreuse yellow
        // Mid-high (0.45 - 0.70): luminous gold yellow -> warm amber orange
        // Peak lesion (0.70 - 1.0): fiery red highlighting active pathology
        final activeColor = getActiveLesionColor(clamped);

        final double effectiveAlpha = (clamped > 0.20)
            ? (alpha * min(1.0, (clamped - 0.20) / 0.45) * 1.6)
            : 0.0;
        final double finalAlpha = effectiveAlpha.clamp(0.0, 0.75);

        final origPixel = base.getPixel(x, y);

        final r = ((1 - finalAlpha) * origPixel.r + finalAlpha * activeColor[0]).round().clamp(0, 255);
        final g = ((1 - finalAlpha) * origPixel.g + finalAlpha * activeColor[1]).round().clamp(0, 255);
        final b = ((1 - finalAlpha) * origPixel.b + finalAlpha * activeColor[2]).round().clamp(0, 255);

        overlayImg.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return Uint8List.fromList(img.encodePng(overlayImg));
  }

  /// Active Clinical Colormap: strictly warm, high-visibility lesion colors.
  /// Completely eliminates blue wash; maps pathology strictly to Green -> Yellow -> Orange -> Fiery Red.
  static List<int> getActiveLesionColor(double v) {
    double r = 0.0, g = 0.0, b = 0.0;
    if (v < 0.20) {
      // Sub-threshold tissue: soft transparent lime
      r = 0.2;
      g = 0.8;
      b = 0.1;
    } else if (v < 0.45) {
      // Transition: Lime Green -> Bright Chartreuse Yellow
      final t = (v - 0.20) / 0.25;
      r = 0.2 + 0.8 * t;
      g = 0.85 + 0.15 * t;
      b = 0.0;
    } else if (v < 0.70) {
      // Transition: Vivid Yellow -> Vibrant Amber Orange
      final t = (v - 0.45) / 0.25;
      r = 1.0;
      g = 1.0 - 0.50 * t;
      b = 0.0;
    } else {
      // Transition: Amber Orange -> Intense Saturated Fiery Red
      final t = ((v - 0.70) / 0.30).clamp(0.0, 1.0);
      r = 1.0;
      g = 0.50 * (1.0 - t);
      b = 0.0;
    }
    return [
      (r * 255).round().clamp(0, 255),
      (g * 255).round().clamp(0, 255),
      (b * 255).round().clamp(0, 255),
    ];
  }
}
