@echo off
REM Skrip ini dijalankan oleh Inno Setup untuk menginstal library Python.
REM Ia mengasumsikan sedang dijalankan dari dalam folder instalasi.

SET "LOG_FILE=%~dp0\pip_install_log.txt"
SET "PYTHON_EXE=%~dp0\python\python.exe"
SET "PIP_EXE=%~dp0\python\Scripts\pip.exe"
SET "WHEELS_DIR=%~dp0\wheels"
SET "REQUIREMENTS_FILE=%~dp0\requirements.txt"

echo [INFO] Log Instalasi Library PDFExtract... > "%LOG_FILE%"
echo. >> "%LOG_FILE%"

echo [INFO] 1. Menginstal Setuptools... >> "%LOG_FILE%"
"%PIP_EXE%" install --no-index --find-links="%WHEELS_DIR%" setuptools >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Gagal menginstal Setuptools. Periksa log di atas. >> "%LOG_FILE%"
    exit /b 1
)

echo. >> "%LOG_FILE%"
echo [INFO] 2. Menginstal requirements.txt... >> "%LOG_FILE%"
"%PIP_EXE%" install --no-index --find-links="%WHEELS_DIR%" -r "%REQUIREMENTS_FILE%" >> "%LOG_FILE%" 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Gagal menginstal requirements.txt. Periksa log di atas. >> "%LOG_FILE%"
    exit /b 1
)

echo. >> "%LOG_FILE%"
echo [INFO] Instalasi library Python selesai dengan sukses. >> "%LOG_FILE%"
exit /b 0