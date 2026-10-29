import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import '../models/history_item.dart';
import '../services/history_service.dart';
import '../utils/app_theme.dart';
import 'results_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      Provider.of<HistoryService>(context, listen: false).loadHistory();
    });
  }

  void _openDetails(DiagnosisHistoryItem item) {
    Uint8List? overlayBytes;
    if (item.gradCamOverlayPath != null &&
        File(item.gradCamOverlayPath!).existsSync()) {
      try {
        overlayBytes = File(item.gradCamOverlayPath!).readAsBytesSync();
      } catch (_) {}
    }

    final result = item.toDiagnosisResult(gradCamOverlayBytes: overlayBytes);

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ResultsScreen(result: result, historyId: item.id),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final historyService = Provider.of<HistoryService>(context);
    final items = historyService.items;

    return Scaffold(
      backgroundColor: AppTheme.background,
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.history_rounded, color: AppTheme.accentCyan, size: 20),
            SizedBox(width: 8),
            Text('Diagnosis History Logs'),
          ],
        ),
        actions: [
          if (items.isNotEmpty)
            IconButton(
              icon: const Icon(
                Icons.delete_sweep_rounded,
                color: AppTheme.textSecondary,
              ),
              tooltip: 'Clear History',
              onPressed: () => _confirmClear(context),
            ),
        ],
      ),
      body:
          items.isEmpty
              ? Center(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 32),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 72,
                        height: 72,
                        decoration: BoxDecoration(
                          color: AppTheme.surfaceElevated,
                          shape: BoxShape.circle,
                          border: Border.all(color: AppTheme.surfaceBorder),
                        ),
                        child: const Icon(
                          Icons.folder_open_rounded,
                          color: AppTheme.textSecondary,
                          size: 36,
                        ),
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'No Previous Diagnoses Yet',
                        style: TextStyle(
                          color: AppTheme.textPrimary,
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'Your detection logs, Grad-CAM heatmaps, and clinical accuracy ratings will appear here automatically.',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 13,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),
              )
              : ListView.separated(
                padding: const EdgeInsets.all(16),
                itemCount: items.length,
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  final item = items[index];
                  return _buildHistoryCard(item);
                },
              ),
    );
  }

  Widget _buildHistoryCard(DiagnosisHistoryItem item) {
    final dateFormat = DateFormat('MMM dd, yyyy • hh:mm a');
    final formattedDate = dateFormat.format(item.timestamp);
    final hasImage = File(item.originalImagePath).existsSync();

    return Material(
      color: AppTheme.surface,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        onTap: () => _openDetails(item),
        borderRadius: BorderRadius.circular(14),
        child: Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.surfaceBorder),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Thumbnail
              ClipRRect(
                borderRadius: BorderRadius.circular(10),
                child: Container(
                  width: 64,
                  height: 64,
                  color: Colors.black26,
                  child:
                      hasImage
                          ? Image.file(
                            File(item.originalImagePath),
                            fit: BoxFit.cover,
                          )
                          : const Icon(
                            Icons.image_not_supported,
                            color: AppTheme.textSecondary,
                          ),
                ),
              ),
              const SizedBox(width: 14),

              // Info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            item.isRecognized
                                ? (item.predictedDisplayName ?? 'DIAGNOSIS')
                                : 'NOT RECOGNISED / NON-TARGET',
                            style: TextStyle(
                              color:
                                  item.isRecognized
                                      ? AppTheme.accentCyan
                                      : AppTheme.dangerRose,
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                        if (item.isRecognized)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 3,
                            ),
                            decoration: BoxDecoration(
                              color: AppTheme.primaryBlue.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              '${(item.confidence * 100).toStringAsFixed(1)}%',
                              style: const TextStyle(
                                color: AppTheme.primaryBlue,
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 4),

                    // Subtype or rejection summary
                    Text(
                      item.clinicalSubtype ??
                          (item.isRecognized
                              ? 'Target Clinical Condition'
                              : 'Out-of-Domain or Look-Alike'),
                      style: const TextStyle(
                        color: AppTheme.textSecondary,
                        fontSize: 12,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 6),

                    // Timestamp and User Verification Status
                    Row(
                      children: [
                        Text(
                          formattedDate,
                          style: const TextStyle(
                            color: AppTheme.textMuted,
                            fontSize: 11,
                          ),
                        ),
                        const Spacer(),
                        if (item.isUserVerifiedCorrect != null)
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 6,
                              vertical: 2,
                            ),
                            decoration: BoxDecoration(
                              color:
                                  item.isUserVerifiedCorrect!
                                      ? AppTheme.successGreen.withOpacity(0.15)
                                      : AppTheme.dangerRose.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(
                                  item.isUserVerifiedCorrect!
                                      ? Icons.check_circle_rounded
                                      : Icons.cancel_rounded,
                                  size: 11,
                                  color:
                                      item.isUserVerifiedCorrect!
                                          ? AppTheme.successGreen
                                          : AppTheme.dangerRose,
                                ),
                                const SizedBox(width: 4),
                                Text(
                                  item.isUserVerifiedCorrect!
                                      ? 'User Verified'
                                      : 'User Marked Incorrect',
                                  style: TextStyle(
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                    color:
                                        item.isUserVerifiedCorrect!
                                            ? AppTheme.successGreen
                                            : AppTheme.dangerRose,
                                  ),
                                ),
                              ],
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              const Icon(
                Icons.chevron_right_rounded,
                color: AppTheme.textSecondary,
                size: 20,
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _confirmClear(BuildContext context) {
    showDialog(
      context: context,
      builder:
          (ctx) => AlertDialog(
            backgroundColor: AppTheme.surfaceElevated,
            title: const Text(
              'Clear History?',
              style: TextStyle(color: AppTheme.textPrimary),
            ),
            content: const Text(
              'This will permanently delete all stored local diagnosis logs and user feedback notes.',
              style: TextStyle(color: AppTheme.textSecondary),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(ctx).pop(),
                child: const Text(
                  'Cancel',
                  style: TextStyle(color: AppTheme.textSecondary),
                ),
              ),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.dangerRose,
                ),
                onPressed: () {
                  Provider.of<HistoryService>(
                    context,
                    listen: false,
                  ).clearAll();
                  Navigator.of(ctx).pop();
                },
                child: const Text(
                  'Clear All',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
    );
  }
}
