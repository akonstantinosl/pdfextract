; Skrip Inno Setup untuk PDFExtract

#define MyAppName "PDFExtract"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#define MyAppPublisher "PDFExtract Developer"
#define MyAppExeName "StartApp.bat"
#define MyPopplerDirName "poppler-25.07.0" ; Pastikan ini sama dengan env di file .yml

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=PDFExtract-Setup-v{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
; Meminta hak Admin, diperlukan untuk instalasi ke Program Files dan menjalankan pip
PrivilegesRequired=admin

[Languages]
; Menyertakan Bahasa Indonesia di wizard
Name: "indonesian"; MessagesFile: "compiler:Languages\Indonesian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Dirs]
; Folder-folder ini akan dibuat di dalam direktori instalasi
Name: "{app}\python"
Name: "{app}\models"
Name: "{app}\poppler"
Name: "{app}\wheels"

[Files]
; Menyalin semua file dari folder 'dist' (yang disiapkan oleh GitHub Actions)
; ke direktori instalasi '{app}' di komputer pengguna.
Source: "dist\python_embed\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs
Source: "dist\models\*"; DestDir: "{app}\models"; Flags: recursesubdirs createallsubdirs
Source: "dist\poppler_bin\{#MyPopplerDirName}\*"; DestDir: "{app}\poppler"; Flags: recursesubdirs createallsubdirs
Source: "dist\wheels\*"; DestDir: "{app}\wheels"; Flags: recursesubdirs createallsubdirs
Source: "dist\main.py"; DestDir: "{app}"
Source: "dist\get-pip.py"; DestDir: "{app}"
Source: "dist\requirements.txt"; DestDir: "{app}"

[Run]
; Ini adalah pengganti 'install.bat'.
; Dijalankan secara otomatis selama proses instalasi wizard.

; 1. Instal Pip
Filename: "{app}\python\python.exe"; Parameters: """{app}\get-pip.py"""; WorkingDir: "{app}"; StatusMsg: "Memasang Pip..."

; 2. Instal semua 'wheels' secara offline
Filename: "{app}\python\Scripts\pip.exe"; Parameters: "install --no-index --find-links=""{app}\wheels"" -r ""{app}\requirements.txt"""; WorkingDir: "{app}"; StatusMsg: "Memasang library Python (Kivy, dll.)... Ini mungkin perlu beberapa saat."

; 3. Buat launcher 'StartApp.bat'
Filename: "{cmd}"; Parameters: "/C echo @echo off > ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo :: Menyiapkan Poppler PATH >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo set ""PATH=%~dp0\poppler\Library\bin;%%PATH%%"" >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo :: Menjalankan aplikasi >> ""{app}\StartApp.bat"""; Flags: runhidden
Filename: "{cmd}"; Parameters: "/C echo start """" ""%~dp0\python\pythonw.exe"" ""%~dp0\main.py"" >> ""{app}\StartApp.bat"""; Flags: runhidden

[Icons]
; Shortcut di Start Menu
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[UninstallDelete]
; Membersihkan semua file saat uninstall
Type: filesandordirs; Name: "{app}"