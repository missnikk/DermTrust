import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../utils/app_theme.dart';

class GradCamViewer extends StatefulWidget {
  final String originalImagePath;
  final Uint8List? gradCamBytes;
  final String diseaseName;

  const GradCamViewer({
    super.key,
    required this.originalImagePath,
    this.gradCamBytes,
    required this.diseaseName,
  });

  @override
  State<GradCamViewer> createState() => _GradCamViewerState();
}

class _GradCamViewerState extends State<GradCamViewer> {
  bool _showHeatmap = true;

  @override
  Widget build(BuildContext context) {
    final hasCam = widget.gradCamBytes != null;

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.surfaceBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header with View Toggle
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Row(
                  children: [
                    Icon(Icons.remove_red_eye_outlined, color: AppTheme.accentCyan, size: 16),
                    SizedBox(width: 6),
                    Text(
                      'EXPLAINABLE AI (GRAD-CAM)',
                      style: TextStyle(
                        color: AppTheme.textPrimary,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ],
                ),
                if (hasCam)
                  Container(
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceElevated,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    padding: const EdgeInsets.all(2),
                    child: Row(
                      children: [
                        _buildToggleButton('Original', !_showHeatmap, () {
                          setState(() => _showHeatmap = false);
                        }),
                        _buildToggleButton('Heatmap', _showHeatmap, () {
                          setState(() => _showHeatmap = true);
                        }),
                      ],
                    ),
                  ),
              ],
            ),
          ),

          // Main Visual Area
          AspectRatio(
            aspectRatio: 1.0,
            child: Container(
              color: Colors.black,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(
                    File(widget.originalImagePath),
                    fit: BoxFit.cover,
                  ),
                  if (hasCam && _showHeatmap)
                    Image.memory(
                      widget.gradCamBytes!,
                      fit: BoxFit.cover,
                    ),
                ],
              ),
            ),
          ),

          // Colormap Gradient Legend
          if (hasCam && _showHeatmap)
            Padding(
              padding: const EdgeInsets.fromLTRB(14, 10, 14, 4),
              child: Column(
                children: [
                  Container(
                    height: 8,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(4),
                      gradient: const LinearGradient(
                        colors: [
                          Color(0xFF0000FF), // Blue
                          Color(0xFF00FFFF), // Cyan
                          Color(0xFF00FF00), // Green
                          Color(0xFFFFFF00), // Yellow
                          Color(0xFFFF0000), // Red
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text('Low Influence (Healthy / BG)', style: TextStyle(color: AppTheme.textMuted, fontSize: 10)),
                      Text('High Activation (Lesion Focus)', style: TextStyle(color: AppTheme.dangerRose, fontSize: 10, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ],
              ),
            ),

          Padding(
            padding: const EdgeInsets.all(12),
            child: Text(
              _showHeatmap && hasCam
                  ? 'Red and yellow regions highlight the morphological features that most strongly steered the model toward the prediction of ${widget.diseaseName}. This visual explanation reflects network activation, not clinical proof of pathology.'
                  : 'Viewing raw optical input image without activation heatmap overlay.',
              style: const TextStyle(
                color: AppTheme.textSecondary,
                fontSize: 12,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleButton(String label, bool isSelected, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: isSelected ? AppTheme.primaryBlue : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.white : AppTheme.textSecondary,
            fontSize: 11,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}
