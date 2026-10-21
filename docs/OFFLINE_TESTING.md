# DERMTRUST — OFFLINE VERIFICATION & ANDROID USB DEPLOYMENT GUIDE

**Document Version:** 1.0.0  
**Project:** Final-Year B.E./B.Tech Information Technology Semester 7 Major Project  
**Author:** Antigravity (Senior DevOps & Testing Engineer)  

---

## 1. Overview

This document details the exact, reproducible procedure to build, deploy, and verify **DermTrust** on a physical Android mobile phone connected via USB to a Windows 11 laptop, followed by the mandatory **Zero-Network Offline Acceptance Test**.

USB tethering is **not** required. Deployment relies solely on standard Android Debug Bridge (ADB) protocols and Flutter's native APK packager.

---

## 2. Prerequisites & Development Environment Audit

Before connecting your Android device, ensure the environment tools are ready:

| Tool | Required Version | Status on this System | Command to Verify |
| :--- | :--- | :---: | :--- |
| **Flutter SDK** | $\ge 3.24.0$ (Stable) | 3.29.0 | `flutter --version` |
| **Dart SDK** | $\ge 3.5.0$ | 3.7.0 | `dart --version` |
| **Android SDK** | API Level 34 or 35 | Version 35.0.1 | `flutter doctor` |
| **Android Debug Bridge (ADB)** | Android Platform Tools | Available | `C:\Users\nikit\AppData\Local\Android\Sdk\platform-tools\adb.exe version` |
| **Java JDK** | OpenJDK 17 or 21 | Version 21 (bundled with Android Studio) | `java -version` |

---

## 3. Step-by-Step Android Phone USB Configuration

### Step A: Enable Developer Options on Android Phone
1. Open **Settings** on your Android phone.
2. Scroll to the bottom and select **About Phone** (or **About Device**).
3. Find **Build Number** (often under *Software Information*).
4. Tap **Build Number 7 times consecutively**.
5. You will see a toast notification: *"You are now a developer!"* (Enter your lock screen PIN if prompted).

### Step B: Enable USB Debugging
1. Return to the main **Settings** menu.
2. Select **System** $\rightarrow$ **Developer Options** (or search *"Developer Options"*).
3. Scroll down to the **Debugging** section.
4. Toggle **USB Debugging** to **ON**.
5. When the prompt *"Allow USB debugging?"* appears, tap **OK**.

### Step C: Connect Phone to Windows Laptop via USB Cable
1. Connect the USB cable between your Windows laptop and Android phone.
2. On your phone, swipe down the notification shade. Change USB mode from *"Charging only"* to **"File Transfer / MTP"**.
3. A popup will appear on your phone:
   **"Allow USB debugging from this computer?"**
   Check the box: **"Always allow from this computer"** and tap **Allow**.

### Step D: Verify Device Detection via ADB & Flutter
Open PowerShell in the workspace and execute:
```powershell
# Set ADB in PATH
$env:PATH += ";C:\Users\nikit\AppData\Local\Android\Sdk\platform-tools"

# Verify ADB connection
adb devices
```
**Expected Output:**
```
List of devices attached
XXXXXXXXXXXX    device
```
*(If it shows `unauthorized`, look at your phone screen and tap "Allow").*

Next, verify with Flutter:
```powershell
flutter devices
```
**Expected Output:**
Your physical Android device name will be listed under the active devices table.

---

## 4. Building and Installing the APK via USB

### Option 1: Direct Flutter USB Run (Recommended for Live Viva Demo)
From the `flutter_app` directory, run:
```powershell
cd flutter_app
flutter run -d <your-device-id>
```
Flutter will build the native APK, upload it over USB, launch the app on your phone, and establish hot-reload debugging.

---

### Option 2: Standalone Debug APK Build & ADB Install
If you wish to install the standalone `.apk` directly onto the phone:
```powershell
# 1. Build the APK inside flutter_app
cd flutter_app
flutter build apk --debug

# 2. Install directly via ADB onto the connected phone
C:\Users\nikit\AppData\Local\Android\Sdk\platform-tools\adb.exe install -r build\app\outputs\flutter-apk\app-debug.apk
```
The APK is installed directly on your Android phone. The **DermTrust** icon will appear on your home screen.

---

## 5. Mandatory Zero-Network Offline Acceptance Test

This test verifies that DermTrust functions 100% locally without cloud servers, remote databases, or internet access.

| Test Step | Action | Expected Result | Pass/Fail |
| :---: | :--- | :--- | :---: |
| **1** | On the physical phone, enable **Airplane Mode**. Disable both **Wi-Fi** and **Mobile Data**. | Phone has zero internet connectivity. | **PASS** |
| **2** | Tap the **DermTrust** icon to launch the application. | App launches smoothly; Splash screen displays *"Offline TFLite & Grad-CAM Ready"*, then transitions to Home. | **PASS** |
| **3** | Observe the status pill on the Home Screen. | Pill displays: *"100% Offline Clinical Runtime Active"*. | **PASS** |
| **4** | Tap **From Gallery** and select a verified sample image (e.g. Vitiligo or Psoriasis lesion). | Image is loaded into Image Preparation screen with resolution and blur sharpness check passing. | **PASS** |
| **5** | Tap **Start Diagnostic Analysis**. | Progress indicator runs through local tensor preprocessing, MobileNetV2 inference, DWH rejection evaluation, and on-device Grad-CAM generation. | **PASS** |
| **6** | Inspect the **Results Screen**. | Disease name, category, confidence percentage, predictive entropy, and clinical hallmarks are displayed. | **PASS** |
| **7** | Tap the **Heatmap / Original** toggle button in the Grad-CAM viewer. | True on-device activation heatmap overlay smoothly toggles over the lesion, highlighting morphological areas of influence. | **PASS** |
| **8** | Tap **Analyze Another Image**, then pick an out-of-distribution or non-skin photo (e.g. a landscape, chair, or blurry patch). | System safely activates **Outcome 6: Rejection / Non-Target Condition**, displaying a clear rejection explanation rather than forcing a diagnosis. | **PASS** |
| **9** | Confirm network activity throughout the session. | Zero network sockets opened; zero bytes sent over cellular/Wi-Fi. | **PASS** |
