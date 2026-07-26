import 'package:flutter/material.dart';

const accent = Color(0xFF34D399);
const warn = Color(0xFFF59E0B);
const danger = Color(0xFFF87171);
const bg0 = Color(0xFF0A1210);
const card = Color(0xFF12211C);
const border = Color(0xFF1F3A30);
const muted = Color(0xFF7E9A8D);
const textColor = Color(0xFFE6F2EC);

ThemeData buildAppTheme() {
  final base = ThemeData.dark();
  return base.copyWith(
    scaffoldBackgroundColor: bg0,
    colorScheme: base.colorScheme.copyWith(
      primary: accent,
      secondary: accent,
      surface: card,
      error: danger,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: bg0,
      foregroundColor: textColor,
      elevation: 0,
    ),
    textTheme: base.textTheme.apply(bodyColor: textColor, displayColor: textColor),
    cardTheme: const CardThemeData(
      color: card,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.all(Radius.circular(14)),
        side: BorderSide(color: border),
      ),
      margin: EdgeInsets.symmetric(vertical: 10),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: accent,
        foregroundColor: const Color(0xFF06231A),
        padding: const EdgeInsets.symmetric(vertical: 14),
        textStyle: const TextStyle(fontWeight: FontWeight.bold),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: const Color(0xFF0F1B17),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(10),
        borderSide: const BorderSide(color: accent),
      ),
    ),
    progressIndicatorTheme: const ProgressIndicatorThemeData(color: accent),
  );
}
