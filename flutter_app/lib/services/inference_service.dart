import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
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

  /// Extracts an asset file into the app's persistent storage to avoid
  /// uncompressed APK mmap alignment issues in tflite_flutter.
  Future<File> _extractAssetToFile(String assetPath) async {
    Directory baseDir;
    try {
      baseDir = await getApplicationSupportDirectory();
    } catch (_) {
      baseDir = await getTemporaryDirectory();
    }

    final modelsDir = Directory('${baseDir.path}/models');
    if (!await modelsDir.exists()) {
      await modelsDir.create(recursive: true);
    }

    final fileName = assetPath.split('/').last;
    final targetFile = File('${modelsDir.path}/$fileName');

    final byteData = await rootBundle.load(assetPath);
    final expectedLength = byteData.lengthInBytes;

    if (await targetFile.exists() && (await targetFile.length()) == expectedLength) {
      return targetFile;
    }

    // Crucial: Use exact offset and length to avoid buffer misalignment
    final bytes = byteData.buffer.asUint8List(
      byteData.offsetInBytes,
      byteData.lengthInBytes,
    );

    final tempFile = File('${modelsDir.path}/$fileName.tmp');
    await tempFile.writeAsBytes(bytes, flush: true);
    if (await targetFile.exists()) {
      await targetFile.delete();
    }
    await tempFile.rename(targetFile.path);

    return targetFile;
  }

  /// Robust multi-strategy interpreter loader:
  /// 1. File-based (page-aligned OS mmap via TfLiteModelCreateFromFile)
  /// 2. Memory buffer with exact byte slice (TfLiteModelCreate)
  /// 3. Standard fromAsset fallback
  Future<Interpreter> _loadInterpreter(String assetPath) async {
    // Strategy 1: Extract model to file and load with options (threads = 2)
    try {
      final file = await _extractAssetToFile(assetPath);
      final options = InterpreterOptions()..threads = 2;
      try {
        final interpreter = Interpreter.fromFile(file, options: options);
        debugPrint('InferenceService: Successfully loaded $assetPath via fromFile (threads=2)');
        return interpreter;
      } catch (e) {
        debugPrint('InferenceService: fromFile with options failed for $assetPath: $e. Retrying without options...');
        final interpreter = Interpreter.fromFile(file);
        debugPrint('InferenceService: Successfully loaded $assetPath via fromFile (default options)');
        return interpreter;
      }
    } catch (e1) {
      debugPrint('InferenceService: File extraction/fromFile failed for $assetPath: $e1. Trying buffer...');
    }

    // Strategy 2: Load via exact buffer slice
    try {
      final byteData = await rootBundle.load(assetPath);
      final bytes = byteData.buffer.asUint8List(
        byteData.offsetInBytes,
        byteData.lengthInBytes,
      );
      final options = InterpreterOptions()..threads = 2;
      try {
        final interpreter = Interpreter.fromBuffer(bytes, options: options);
        debugPrint('InferenceService: Successfully loaded $assetPath via fromBuffer (threads=2)');
        return interpreter;
      } catch (e) {
        final interpreter = Interpreter.fromBuffer(bytes);
        debugPrint('InferenceService: Successfully loaded $assetPath via fromBuffer (default options)');
        return interpreter;
      }
    } catch (e2) {
      debugPrint('InferenceService: fromBuffer failed for $assetPath: $e2. Retrying via fromAsset...');
    }

    // Strategy 3: fromAsset fallback
    final options = InterpreterOptions()..threads = 2;
    return await Interpreter.fromAsset(assetPath, options: options);
  }

  /// Loads TFLite models and JSON metadata from app assets into memory.
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // 1. Load Metadata
      final metaStr = await rootBundle.loadString(AppConstants.metadataAsset);
      _metadata = json.decode(metaStr) as Map<String, dynamic>;
      _modelClasses = List<String>.from(
        _metadata?['classes'] ?? ['acne', 'melasma', 'psoriasis', 'ringworm', 'vitiligo'],
      );

      // 2. Load CAM Weights
      try {
        final weightsStr = await rootBundle.loadString(AppConstants.camWeightsAsset);
        _camWeightsData = json.decode(weightsStr);
      } catch (e) {
        debugPrint('InferenceService: Warning loading CAM weights: $e');
      }

      // 3. Load Primary Classifier Interpreter
      _classifierInterpreter = await _loadInterpreter(AppConstants.classifierModelAsset);

      // 4. Load CAM Interpreter (graceful degradation if CAM model fails to load)
      try {
        _camInterpreter = await _loadInterpreter(AppConstants.camModelAsset);
      } catch (e) {
        debugPrint('InferenceService: Grad-CAM interpreter unavailable: $e. Proceeding with classifier only.');
        _camInterpreter = null;
      }

      _isInitialized = true;
    } catch (e) {
      _isInitialized = false;
      debugPrint('InferenceService initialization failed: $e');
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
    if (_camInterpreter != null) {
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
        camOutputs[probIdx] = List<List<double>>.generate(
          1,
          (_) => List<double>.filled(numClasses, 0.0),
        );
        camOutputs[featIdx] = List<List<List<List<double>>>>.generate(
          1,
          (_) => List<List<List<double>>>.generate(
            7,
            (_) => List<List<double>>.generate(
              7,
              (_) => List<double>.filled(1280, 0.0),
            ),
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
      } catch (e) {
        debugPrint('InferenceService: CAM inference failed: $e');
        // Fallback: Continue without overlay if CAM pass fails
        overlayBytes = null;
      }
    }

    // Clinical note: The model is a 5-class categorical classifier and does not predict
    // disease severity or inflammatory staging. Subtype evaluation requires clinical exam.
    const String? severity = null;

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
