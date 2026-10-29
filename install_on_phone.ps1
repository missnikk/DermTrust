# DermTrust - One-Click APK Installer and Phone Launcher
$AdbPath = "C:\Users\nikit\AppData\Local\Android\Sdk\platform-tools\adb.exe"
$PackageName = "com.vesit.dermtrust.dermtrust"

# Identify the release APK path
$ReleaseApk = "$PSScriptRoot\releases\app-release.apk"
$BuildApk = "$PSScriptRoot\flutter_app\build\app\outputs\flutter-apk\app-release.apk"

if (Test-Path $ReleaseApk) {
    $ApkPath = $ReleaseApk
} elseif (Test-Path $BuildApk) {
    $ApkPath = $BuildApk
} else {
    Write-Host "[ERROR] No release APK found at $ReleaseApk or $BuildApk" -ForegroundColor Red
    exit 1
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "DermTrust Phone Deployment Tool (Production Prototype)" -ForegroundColor Cyan
Write-Host "Target APK: $ApkPath" -ForegroundColor Gray
Write-Host "Package:    $PackageName" -ForegroundColor Gray
Write-Host "==================================================" -ForegroundColor Cyan

Write-Host "Checking for connected Android devices via ADB..." -ForegroundColor Yellow
$devices = & $AdbPath devices
Write-Host $devices

if ($devices -match "device\b") {
    Write-Host "`n[OK] Android device detected. Installing APK..." -ForegroundColor Green
    & $AdbPath install -r $ApkPath
    Write-Host "`nLaunching DermTrust on phone..." -ForegroundColor Green
    & $AdbPath shell am start -n "$PackageName/.MainActivity"
    Write-Host "`n[SUCCESS] DermTrust launched successfully!" -ForegroundColor Green
} else {
    Write-Host "`n[NOTICE] No authorized Android device detected on ADB." -ForegroundColor Red
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  1. Unplug and replug the USB cable."
    Write-Host "  2. Unlock your phone screen."
    Write-Host "  3. When prompted 'Allow USB debugging?', tap Allow."
    Write-Host "  4. Set USB mode to 'File Transfer / Android Auto'."
}
