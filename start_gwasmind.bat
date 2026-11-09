@echo off
setlocal
chcp 65001 >nul

set "GWASMIND_ROOT=%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%GWASMIND_ROOT%start_gwasmind.ps1" %*
set "GWASMIND_EXIT=%ERRORLEVEL%"

if not "%GWASMIND_EXIT%"=="0" (
  echo.
  echo GWASMind launcher exited with code %GWASMIND_EXIT%.
)

exit /b %GWASMIND_EXIT%
