import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import '../models/diagnosis_result.dart';
import '../models/history_item.dart';

class HistoryService extends ChangeNotifier {
  static final HistoryService _instance = HistoryService._internal();
  factory HistoryService() => _instance;
  HistoryService._internal();

  List<DiagnosisHistoryItem> _items = [];
  bool _isLoaded = false;

  List<DiagnosisHistoryItem> get items => List.unmodifiable(_items);

  Future<File> _getHistoryFile() async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/diagnosis_history.json');
  }

  Future<void> loadHistory() async {
    if (_isLoaded) return;
    try {
      final file = await _getHistoryFile();
      if (await file.exists()) {
        final content = await file.readAsString();
        if (content.isNotEmpty) {
          final List<dynamic> jsonList = jsonDecode(content);
          _items =
              jsonList.map((j) => DiagnosisHistoryItem.fromJson(j)).toList();
          // Sort newest first
          _items.sort((a, b) => b.timestamp.compareTo(a.timestamp));

          debugPrint('DERMTRUST_HISTORY_COUNT: ${_items.length}');
          for (final item in _items) {
            debugPrint(
              'DERMTRUST_HISTORY_RECORD: ${jsonEncode(item.toJson())}',
            );
          }

          try {
            final ext = await getExternalStorageDirectory();
            if (ext != null) {
              final exportFile = File(
                '${ext.path}/dermtrust_history_export.json',
              );
              await exportFile.writeAsString(content);
            }
            final dlDir = Directory('/sdcard/Download');
            if (await dlDir.exists()) {
              final dlFile = File(
                '/sdcard/Download/dermtrust_history_export.json',
              );
              await dlFile.writeAsString(content);
            }
          } catch (_) {}
        }
      }
      _isLoaded = true;
      notifyListeners();
    } catch (e) {
      debugPrint('HistoryService: Failed to load history: $e');
    }
  }

  Future<void> _persistHistory() async {
    try {
      final file = await _getHistoryFile();
      final jsonList = _items.map((i) => i.toJson()).toList();
      final str = jsonEncode(jsonList);
      await file.writeAsString(str);

      debugPrint('DERMTRUST_HISTORY_SAVED: ${_items.length} records');
      for (final item in _items) {
        debugPrint('DERMTRUST_HISTORY_ENTRY: ${jsonEncode(item.toJson())}');
      }

      try {
        final ext = await getExternalStorageDirectory();
        if (ext != null) {
          final exportFile = File('${ext.path}/dermtrust_history_export.json');
          await exportFile.writeAsString(str);
        }
        final dlDir = Directory('/sdcard/Download');
        if (await dlDir.exists()) {
          final dlFile = File('/sdcard/Download/dermtrust_history_export.json');
          await dlFile.writeAsString(str);
        }
      } catch (_) {}
    } catch (e) {
      debugPrint('HistoryService: Failed to save history to disk: $e');
    }
  }

  Future<DiagnosisHistoryItem> addRecord(DiagnosisResult result) async {
    await loadHistory();

    final id = DateTime.now().millisecondsSinceEpoch.toString();

    // Optionally save Grad-CAM overlay to file if bytes exist
    String? overlayPath;
    if (result.gradCamOverlayBytes != null) {
      try {
        final dir = await getApplicationDocumentsDirectory();
        final overlayFile = File('${dir.path}/overlay_$id.png');
        await overlayFile.writeAsBytes(result.gradCamOverlayBytes!);
        overlayPath = overlayFile.path;
      } catch (_) {}
    }

    final item = DiagnosisHistoryItem(
      id: id,
      timestamp: result.timestamp,
      originalImagePath: result.originalImagePath,
      gradCamOverlayPath: overlayPath,
      isRecognized: result.isRecognized,
      predictedDiseaseId: result.predictedDiseaseId,
      predictedDisplayName: result.predictedDisplayName,
      confidence: result.confidence,
      entropy: result.entropy,
      classProbabilities: result.classProbabilities,
      detectedSkinTone: result.detectedSkinTone,
      itaAngle: result.itaAngle,
      acneSeverity: result.acneSeverity,
      clinicalSubtype: result.clinicalSubtype,
      rejectionReason: result.rejectionReason,
    );

    _items.insert(0, item);
    notifyListeners();
    await _persistHistory();
    return item;
  }

  Future<void> updateFeedback({
    required String historyId,
    required bool isCorrect,
    String? userReportedActualDisease,
    String? feedbackNotes,
  }) async {
    await loadHistory();

    final idx = _items.indexWhere((i) => i.id == historyId);
    if (idx != -1) {
      _items[idx] = _items[idx].copyWithFeedback(
        isUserVerifiedCorrect: isCorrect,
        userReportedActualDisease: userReportedActualDisease,
        userFeedbackNotes: feedbackNotes,
      );
      notifyListeners();
      await _persistHistory();
    }
  }

  Future<void> clearAll() async {
    _items.clear();
    notifyListeners();
    try {
      final file = await _getHistoryFile();
      if (await file.exists()) {
        await file.delete();
      }
    } catch (_) {}
  }
}
