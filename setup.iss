; Skrip Inno Setup untuk PDFExtract

; --- Definisi Variabel Global ---
#define MyAppName "PDFExtract"
; MyAppVersion diambil dari GitHub Actions (/DMyAppVersion=...)
#ifndef MyAppVersion
  ; Versi default jika kompilasi manual
  #define MyAppVersion "0.0.0-dev"
#endif
#define MyAppPublisher "JST Indonesia"
; Menjalankan pythonw.exe sebagai launcher utama
#define MyAppExeName "pythonw.exe" 
#define MyPopplerDirName "poppler-25.07.0"

[Setup]
; Informasi dasar aplikasi
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName}
UninstallDisplayName={#MyAppName}

; Lokasi instalasi (di dalam Program Files)
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}

; Ikon untuk installer (diambil dari variabel GitHub Actions)
SetupIconFile=dist\pdfextract.ico
; Ikon untuk Uninstaller di Control Panel
UninstallDisplayIcon={app}\pdfextract.ico
; Nama file output installer
OutputBaseFilename=PDFExtract-Setup-v{#MyAppVersion}

; Pengaturan kompresi untuk installer yang lebih kecil
Compression=lzma
SolidCompression=yes
WizardStyle=modern

; Meminta hak Admin, diperlukan untuk instalasi ke Program Files
PrivilegesRequired=admin

[Languages]
; Menentukan bahasa yang tersedia saat instalasi
Name: "indonesian"; MessagesFile: "compiler:Languages\Indonesian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Messages]
; Kustomisasi teks UI untuk bahasa Indonesia
indonesian.FinishedHeading=Instalasi Selesai
indonesian.FinishedLabel=Setup telah selesai menginstal {#MyAppName} di komputer Anda. Aplikasi dapat dijalankan dengan memilih shortcut yang terpasang.
indonesian.RunLabel=Luncurkan {#MyAppName}

; Kustomisasi teks UI untuk bahasa Inggris
english.FinishedHeading=Setup Complete
english.FinishedLabel=Setup has finished installing {#MyAppName} on your computer. The application may be launched by selecting the installed shortcuts.
english.RunLabel=Launch {#MyAppName}

[Dirs]
; Direktori kustom yang akan dibuat di dalam folder instalasi
Name: "{app}\python"
Name: "{app}\models"
Name: "{app}\poppler"
Name: "{app}\wheels"

[Files]
; Menyalin semua file dari folder 'dist' (staging) ke folder instalasi
Source: "dist\python_embed\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs
Source: "dist\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs
Source: "dist\poppler_bin\{#MyPopplerDirName}\*"; DestDir: "{app}\poppler"; Flags: recursesubdirs createallsubdirs
Source: "dist\wheels\*"; DestDir: "{app}\wheels"; Flags: recursesubdirs createallsubdirs
Source: "dist\main.py"; DestDir: "{app}"
Source: "dist\requirements.txt"; DestDir: "{app}"
Source: "dist\install_libs.bat"; DestDir: "{app}"
Source: "dist\pdfextract.ico"; DestDir: "{app}"
Source: "dist\get-pip.py"; DestDir: "{app}"
Source: "dist\vc_redist.x64.exe"; DestDir: "{app}"; Flags: deleteafterinstall

[Run]
; 1. Memasang Microsoft Visual C++ Redistributable
Filename: "{app}\vc_redist.x64.exe"; \
  Parameters: "/install /passive /norestart"; \
  WorkingDir: "{app}"; \
  StatusMsg: "Memasang komponen sistem Microsoft Visual C++..."; \
  Flags: runhidden skipifdoesntexist waituntilterminated
; 2. Memasang Pip
;    Menjalankan get-pip.py secara offline menggunakan wheels yang sudah di-bundle
Filename: "{app}\python\python.exe"; \
  Parameters: """{app}\get-pip.py"" --no-index --find-links=""{app}\wheels"""; \
  WorkingDir: "{app}"; \
  StatusMsg: "Memasang Pip..."; \
  Flags: runhidden

; 3. Menjalankan batch script untuk menginstal library dari folder 'wheels'
Filename: "{app}\install_libs.bat"; \
  WorkingDir: "{app}"; \
  StatusMsg: "Memasang library Python... Ini mungkin perlu beberapa saat."; \
  Flags: runhidden

; 4. (Opsional) Menjalankan aplikasi setelah instalasi selesai jika user mencentang box
Filename: "{app}\python\{#MyAppExeName}"; \
  Parameters: """{app}\main.py"""; \
  WorkingDir: "{app}"; \
  Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent

[Icons]
; Membuat shortcut di Start Menu
; Target: pythonw.exe
; Argumen: main.py
; Mulai di: folder aplikasi
; Ikon: logo aplikasi
Name: "{group}\{#MyAppName}"; \
  Filename: "{app}\python\{#MyAppExeName}"; \
  Parameters: """{app}\main.py"""; \
  WorkingDir: "{app}"; \
  IconFilename: "{app}\pdfextract.ico"

; Membuat shortcut Uninstaller di Start Menu
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
; Membersihkan seluruh folder aplikasi saat uninstall
Type: filesandordirs; Name: "{app}"