; Skrip Inno Setup untuk PDFExtract

#define MyAppName "PDFExtract"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev" // Versi diambil dari tag Git
#endif
#define MyAppPublisher "PDFExtract Developer"
#define MyAppExeName "StartApp.bat"
#define MyPopplerDirName "poppler-25.07.0"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName} ; Lokasi instalasi default
DefaultGroupName={#MyAppName} ; Nama folder Start Menu default
SetupIconFile="{#SetupIconAbsPath}" ; Logo untuk file Setup.exe (Path absolut dari workflow)
UninstallDisplayIcon={app}\pdfextract.png ; Logo untuk Uninstaller di Control Panel (Path saat runtime)
OutputBaseFilename=PDFExtract-Setup-v{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
 ; Meminta hak Admin, diperlukan untuk instalasi ke Program Files dan menjalankan pip
 PrivilegesRequired=admin

[Languages]
Name: "indonesian"; MessagesFile: "compiler:Languages\Indonesian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

; --- (OPSIONAL) Pesan Kustom untuk Wizard ---
[Messages]
Indonesian.FinishedHeading=Instalasi Selesai
Indonesian.FinishedLabel=Setup telah selesai menginstal {#MyAppName} di komputer Anda. Aplikasi dapat dijalankan dengan memilih shortcut yang terpasang.
Indonesian.RunLabel=Luncurkan {#MyAppName} ; Teks untuk checkbox di halaman terakhir

English.FinishedHeading=Setup Complete
English.FinishedLabel=Setup has finished installing {#MyAppName} on your computer. The application may be launched by selecting the installed shortcuts.
English.RunLabel=Launch {#MyAppName} ; Text for the checkbox on the final page
; --- Akhir Pesan Kustom ---

[Dirs]
Name: "{app}\python"
Name: "{app}\models"
Name: "{app}\poppler"
Name: "{app}\wheels"

[Files]
; Menyalin semua file dari folder 'dist'
Source: "dist\python_embed\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs
Source: "dist\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs
Source: "dist\poppler_bin\{#MyPopplerDirName}\*"; DestDir: "{app}\poppler"; Flags: recursesubdirs createallsubdirs
Source: "dist\wheels\*"; DestDir: "{app}\wheels"; Flags: recursesubdirs createallsubdirs
Source: "dist\main.py"; DestDir: "{app}"
Source: "dist\get-pip.py"; DestDir: "{app}"
Source: "dist\requirements.txt"; DestDir: "{app}"
Source: "dist\install_libs.bat"; DestDir: "{app}"
Source: "dist\pdfextract.png"; DestDir: "{app}" ; <-- Salin logo ke folder instalasi

[Run]
; Filename: "{app}\vc_redist.x64.exe"; Parameters: ... (HAPUS BARIS INI)

; 1. Instal Pip (Sekarang jadi langkah pertama)
Filename: "{app}\python\python.exe"; Parameters: """{app}\get-pip.py"""; WorkingDir: "{app}"; StatusMsg: "Memasang Pip..."

; 2. Jalankan skrip batch instalasi library
Filename: "{app}\install_libs.bat"; WorkingDir: "{app}"; StatusMsg: "Memasang library Python (KivyMD, Pandas, dll.)... Ini mungkin perlu beberapa saat."

; 3. Buat launcher 'StartApp.bat'
Filename: "{cmd}"; Parameters: "/C echo @echo off > ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo :: Menyiapkan Poppler PATH >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo set ""PATH=%~dp0\poppler\Library\bin;%%PATH%%"" >> ""{app}\StartApp.bat"""; Flags: runhidden
; Baris untuk cek versi Python (opsional tapi bagus untuk debug)
Filename: "{cmd}"; Parameters: "/C echo :: Cek versi Python untuk debug >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo ""%~dp0\python\python.exe"" --version >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo. >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo :: Menjalankan aplikasi >> ""{app}\StartApp.bat"""; Flags: runhidden
; Tetap python.exe untuk debug
Filename: "{cmd}"; Parameters: "/C echo ""%~dp0\python\python.exe"" ""%~dp0\main.py"" >> ""{app}\StartApp.bat"""; Flags: runhidden
; Tetap pause untuk debug
Filename: "{cmd}"; Parameters: "/C echo pause >> ""{app}\StartApp.bat"""; Flags: runhidden

[Icons]
; Shortcut di Start Menu
; Menambahkan IconFilename dan Flags: run
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\pdfextract.png"; Flags: run
; Shortcut Uninstaller (otomatis dapat ikon dari Setup section)
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
; Membersihkan semua file saat uninstall
Type: filesandordirs; Name: "{app}"