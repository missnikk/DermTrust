import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../models/diagnosis_result.dart';
import '../models/disease_info.dart';
import '../utils/constants.dart';
import 'image_processor.dart';

class InferenceService {
  Interpreter? _classifierInterpreter;
  Interpreter? _camInterpreter;
  Map<String, dynamic>? _metadata;
  dynamic _camWeightsData;
  List<String> _modelClasses = [];

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  /// Loads TFLite models and JSON metadata from app assets into memory.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // 1. Load Metadata
      final metaStr = await rootBundle.loadString(AppConstants.metadataAsset);
      _metadata = json.decode(metaStr) as Map<String, dynamic>;
      _modelClasses = List<String>.from(_metadata?['classes'] ?? ['acne', 'psoriasis', 'vitiligo']);

      // 2. Load CAM Weights
      final weightsStr = await rootBundle.loadString(AppConstants.camWeightsAsset);
      _camWeightsData = json.decode(weightsStr);

      // 3. Load TFLite Interpreters with CPU options
      final options = InterpreterOptions()..threads = 2;

      _classifierInterpreter = await Interpreter.fromAsset(
        AppConstants.classifierModelAsset,
        options: options,
      );

      _camInterpreter = await Interpreter.fromAsset(
        AppConstants.camModelAsset,
        options: options,
      );

      _isInitialized = true;
    } catch (e) {
      // Allow fallback if asset loading encounters edge condition
      _isInitialized = false;
      rethrow;
    }
  }

  /// Evaluates Shannon Entropy in bits
  double _computeEntropy(List<double> probs) {
    double entropy = 0.0;
    const eps = 1e-9;
    for (final p in probs) {
      final clipped = max(eps, min(1.0, p));
      entropy -= clipped * (log(clipped) / ln2);
    }
    return entropy;
  }

  /// Full end-to-end Dermatological Workflow Hierarchy (DWH) execution.
  Future<DiagnosisResult> analyzeImage(File imageFile) async {
    if (!_isInitialized) {
      await initialize();
    }

    // Tier 1: Input Validation (Format, Decoding, Resolution, Blur)
    final validation = await ImageProcessor.validateImage(imageFile);
    if (!validation.isValid) {
      return DiagnosisResult.rejection(
        reason: validation.errorMessage ?? AppConstants.errBlurryImage,
        originalImagePath: imageFile.path,
      );
    }

    final decodedImg = validation.decodedImage!;
    final inputFloat32 = ImageProcessor.preprocessForModel(decodedImg);

    // Prepare TFLite tensors: [1, 224, 224, 3]
    final inputTensor = inputFloat32.reshape([1, 224, 224, 3]);

    final numClasses = _modelClasses.length;
    final outputProbs = List<List<double>>.filled(
      1,
      List<double>.filled(numClasses, 0.0),
    );

    // Run Primary Classifier
    _classifierInterpreter!.run(inputTensor, outputProbs);
    final rawProbs = outputProbs[0];

    // Find predicted class index and confidence
    int maxIdx = 0;
    double maxConf = rawProbs[0];
    for (int i = 1; i < numClasses; i++) {
      if (rawProbs[i] > maxConf) {
        maxConf = rawProbs[i];
        maxIdx = i;
      }
    }

    final entropy = _computeEntropy(rawProbs);
    final Map<String, double> probMap = {};
    for (int i = 0; i < numClasses; i++) {
      probMap[_modelClasses[i]] = rawProbs[i];
    }

    final predictedId = _modelClasses[maxIdx];
    final diseaseInfo = DiseaseInfo.getById(predictedId);
    final displayName = diseaseInfo?.displayName ?? predictedId.toUpperCase();

    // Tier 2: DWH Rejection Gate (Calibrated Confidence + Predictive Entropy)
    if (maxConf < AppConstants.minConfidenceThreshold) {
      return DiagnosisResult.rejection(
        reason: AppConstants.errLowConfidence,
        originalImagePath: imageFile.path,
        confidence: maxConf,
        entropy: entropy,
        classProbabilities: probMap,
      );
    }

    if (entropy > AppConstants.maxEntropyBits) {
      return DiagnosisResult.rejection(
        reason: AppConstants.errHighEntropy,
        originalImagePath: imageFile.path,
        confidence: maxConf,
        entropy: entropy,
        classProbabilities: probMap,
      );
    }

    // Tier 3 & 4: On-Device Grad-CAM Generation
    Uint8List? overlayBytes;
    try {
      // Dynamically identify feature map tensor [1, 7, 7, 1280] vs probability tensor [1, numClasses]
      int featIdx = 0;
      int probIdx = 1;
      try {
        final outTensors = _camInterpreter!.getOutputTensors();
        for (int i = 0; i < outTensors.length; i++) {
          if (outTensors[i].shape.length == 4) {
            featIdx = i;
          } else if (outTensors[i].shape.length == 2) {
            probIdx = i;
          }
        }
      } catch (_) {}

      final camOutputs = <int, Object>{};
      camOutputs[probIdx] = List<List<double>>.filled(1, List<double>.filled(numClasses, 0.0));
      camOutputs[featIdx] = List<List<List<List<double>>>>.filled(
        1,
        List<List<List<double>>>.filled(
          7,
          List<List<double>>.filled(7, List<double>.filled(1280, 0.0)),
        ),
      );

      _camInterpreter!.runForMultipleInputs([inputTensor], camOutputs);
      final dynamic featureMap = (camOutputs[featIdx] as dynamic)[0]; // [7, 7, 1280]

      // Extract class weights
      List<double> weights = [];
      if (_camWeightsData is Map && _camWeightsData.containsKey('weights')) {
        final allWeights = _camWeightsData['weights'] as List;
        weights = List<double>.from(allWeights[maxIdx]);
      } else if (_camWeightsData is Map && _camWeightsData.containsKey(predictedId)) {
        weights = List<double>.from(_camWeightsData[predictedId]);
      }

      // Linear combination: CAM = ReLU(sum_k w_k * F_k)
      final cam7x7 = List<List<double>>.generate(
        7,
        (y) => List<double>.generate(7, (x) {
          double sum = 0.0;
          for (int k = 0; k < min(weights.length, 1280); k++) {
            final double featVal = (featureMap[y][x][k] as num).toDouble();
            sum += weights[k] * featVal;
          }
          return max(0.0, sum); // ReLU
        }),
      );

      // Normalize 7x7 to [0, 1]
      double minVal = double.infinity;
      double maxVal = -double.infinity;
      for (final row in cam7x7) {
        for (final v in row) {
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
      }

      final range = (maxVal - minVal > 1e-6) ? (maxVal - minVal) : 1.0;
      final normCam7x7 = cam7x7
          .map((row) => row.map((v) => (v - minVal) / range).toList())
          .toList();

      overlayBytes = ImageProcessor.createGradCamOverlay(
        original: decodedImg,
        heatmap7x7: normCam7x7,
      );
    } catch (_) {
      // Fallback: Continue without overlay if CAM pass fails
      overlayBytes = null;
    }

    // Acne Severity Stage (if Acne detected)
    String? severity;
    if (predictedId == 'acne') {
      if (maxConf > 0.85) {
        severity = 'Moderate to Severe (High Inflammatory Activation)';
      } else if (maxConf > 0.70) {
        severity = 'Mild to Moderate (Papular/Comedonal)';
      } else {
        severity = 'Mild / Early-Stage';
      }
    }

    return DiagnosisResult(
      isRecognized: true,
      predictedDiseaseId: predictedId,
      predictedDisplayName: displayName,
      confidence: maxConf,
      entropy: entropy,
      classProbabilities: probMap,
      originalImagePath: imageFile.path,
      gradCamOverlayBytes: overlayBytes,
      acneSeverity: severity,
      timestamp: DateTime.now(),
    );
  }

  void dispose() {
    _classifierInterpreter?.close();
    _camInterpreter?.close();
    _isInitialized = false;
  }
}
