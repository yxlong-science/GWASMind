; Inno Setup script for the GWASMind Windows installer.
; Build with packaging\build_installer.ps1, which stages the files and passes
; AppVersion, NumericVersion, StageDir, OutputDir and SourceRoot.

#ifndef AppVersion
  #error AppVersion must be defined (use build_installer.ps1)
#endif

[Setup]
AppId={{846F14C9-2C18-49BC-8364-364C09177CBF}
AppName=GWASMind
AppVersion={#AppVersion}
AppVerName=GWASMind {#AppVersion}
AppPublisher=GWASMind contributors
VersionInfoVersion={#NumericVersion}
; Analyses, history and saved settings are written inside the installation
; folder, so it must be writable by the user: install per user, without admin.
PrivilegesRequired=lowest
DefaultDirName={localappdata}\GWASMind
DisableProgramGroupPage=yes
DefaultGroupName=GWASMind
LicenseFile={#SourceRoot}\LICENSE
SetupIconFile={#SourceRoot}\packaging\gwasmind.ico
UninstallDisplayIcon={app}\GWASMind.exe
UninstallDisplayName=GWASMind {#AppVersion}
OutputDir={#OutputDir}
OutputBaseFilename=GWASMind-{#AppVersion}-Setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; Replace program files on upgrade so no stale modules from an older version
; remain. User data (artifacts, data, .gwasmind) is never touched here.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\src"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\GWASMind"; Filename: "{app}\GWASMind.exe"; WorkingDir: "{app}"; Comment: "Start the local GWASMind service and open it in the browser"
Name: "{autodesktop}\GWASMind"; Filename: "{app}\GWASMind.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\GWASMind.exe"; Description: "{cm:LaunchProgram,GWASMind}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\src"

[Code]
function LongPathsEnabled(): Boolean;
var
  Value: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM, 'SYSTEM\CurrentControlSet\Control\FileSystem',
    'LongPathsEnabled', Value) and (Value = 1);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and not LongPathsEnabled() then
    SuppressibleMsgBox('Windows long-path support is disabled on this computer.' + #13#10 + #13#10 +
      'GWASMind saves analyses in nested folders that can exceed 260 characters. ' +
      'Ask an administrator to run this once in PowerShell (Run as administrator):' + #13#10 + #13#10 +
      'New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" ' +
      '-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force',
      mbInformation, MB_OK, IDOK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  if not (DirExists(ExpandConstant('{app}\artifacts')) or DirExists(ExpandConstant('{app}\data'))
      or DirExists(ExpandConstant('{app}\.gwasmind'))) then
    exit;
  if SuppressibleMsgBox('Also delete your GWASMind analyses, history and saved model settings in' + #13#10 +
      ExpandConstant('{app}') + '?' + #13#10 + #13#10 +
      'Choose No to keep them for a later installation.',
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
  begin
    DelTree(ExpandConstant('{app}\artifacts'), True, True, True);
    DelTree(ExpandConstant('{app}\data'), True, True, True);
    DelTree(ExpandConstant('{app}\.gwasmind'), True, True, True);
    RemoveDir(ExpandConstant('{app}'));
  end;
end;
