import 'package:flutter/material.dart';

class AppTheme {
  // Medical Blue Family Palette
  static const Color background = Color(0xFF0A0F1D); // Deep clinical slate navy
  static const Color surface = Color(0xFF111A2E); // Medical blue-tinted surface
  static const Color surfaceElevated = Color(
    0xFF17233D,
  ); // Elevated card surface
  static const Color surfaceBorder = Color(0xFF233554); // Subtle blue border

  // Primary Clinical Blue (Take Photo, Key Accents, Branding)
  static const Color primaryBlue = Color(
    0xFF2563EB,
  ); // Cobalt / Royal Clinical Blue
  static const Color primaryBlueDark = Color(0xFF1D4ED8); // Deep Sapphire Blue

  // Secondary Harmonious Blue (Gallery, Chips, Accents - strictly from Blue family)
  static const Color accentCyan = Color(
    0xFF0284C7,
  ); // Cerulean / Ocean Blue (Blue family)
  static const Color accentBlueLight = Color(0xFF38BDF8); // Ice Blue accent
  static const Color galleryBlueSurface = Color(
    0xFF132238,
  ); // Refined dark blue for gallery button
  static const Color galleryBlueBorder = Color(
    0xFF2563EB,
  ); // Subtle blue outline

  // Clinical Status Indicators
  static const Color successGreen = Color(0xFF10B981);
  static const Color warningAmber = Color(0xFFF59E0B);
  static const Color dangerRose = Color(0xFFEF4444);

  // High-legibility Clinical Typography
  static const Color textPrimary = Color(0xFFF8FAFC);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textMuted = Color(0xFF64748B);

  static ThemeData get darkTheme {
    return ThemeData(
      brightness: Brightness.dark,
      scaffoldBackgroundColor: background,
      primaryColor: primaryBlue,
      canvasColor: surface,
      cardColor: surface,
      dividerColor: surfaceBorder,
      colorScheme: const ColorScheme.dark(
        primary: primaryBlue,
        secondary: accentCyan,
        surface: surface,
        error: dangerRose,
        onPrimary: Colors.white,
        onSurface: textPrimary,
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: background,
        elevation: 0,
        centerTitle: true,
        titleTextStyle: TextStyle(
          color: textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.5,
        ),
        iconTheme: IconThemeData(color: textPrimary),
      ),
      cardTheme: CardTheme(
        color: surface,
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(14),
          side: const BorderSide(color: surfaceBorder, width: 1),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryBlue,
          foregroundColor: Colors.white,
          elevation: 2,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: textPrimary,
          side: const BorderSide(color: surfaceBorder),
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
          ),
        ),
      ),
      textTheme: const TextTheme(
        displayLarge: TextStyle(
          color: textPrimary,
          fontSize: 26,
          fontWeight: FontWeight.bold,
        ),
        displayMedium: TextStyle(
          color: textPrimary,
          fontSize: 22,
          fontWeight: FontWeight.bold,
        ),
        titleLarge: TextStyle(
          color: textPrimary,
          fontSize: 18,
          fontWeight: FontWeight.w600,
        ),
        titleMedium: TextStyle(
          color: textPrimary,
          fontSize: 15,
          fontWeight: FontWeight.w600,
        ),
        bodyLarge: TextStyle(color: textPrimary, fontSize: 14, height: 1.5),
        bodyMedium: TextStyle(color: textSecondary, fontSize: 13, height: 1.4),
        bodySmall: TextStyle(color: textMuted, fontSize: 11),
      ),
    );
  }
}
