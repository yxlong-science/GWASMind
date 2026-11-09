[CmdletBinding()]
param(
    # Official CPython x64 runtime from the "python" NuGet package (python.org).
    [string]$PythonVersion = "3.13.17",
    # Optional: an existing full CPython 3.11+ x64 directory to bundle instead.
    [string]$PythonHome = "",
    # Inno Setup command-line compiler; found on PATH or in the usual folders if omitted.
    [string]$InnoCompiler = "",
    [string]$OutputDir = ""
)
# Builds dist\GWASMind-<version>-Setup.exe: a per-user Windows installer that
# bundles a private Python runtime, GWASMind's dependencies and its source tree.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Packaging = $PSScriptRoot
$Root = Split-Path -Parent $Packaging
$Build = Join-Path $Packaging "build"
$Stage = Join-Path $Build "stage"
if (-not $OutputDir) { $OutputDir = Join-Path $Root "dist" }

function Invoke-Checked {
    param([string]$FilePath, [string[]]$ArgumentList)
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) { throw "$FilePath failed with exit code $LASTEXITCODE." }
}

function Find-InnoCompiler {
    if ($InnoCompiler) { return $InnoCompiler }
    $command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    foreach ($candidate in @(
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    throw "Inno Setup 6 (ISCC.exe) was not found; install it from https://jrsoftware.org or pass -InnoCompiler."
}

function Get-PythonRuntime {
    if ($PythonHome) { return (Resolve-Path -LiteralPath $PythonHome).Path }
    $cache = Join-Path $Build "python-$PythonVersion"
    $tools = Join-Path $cache "tools"
    if (-not (Test-Path -LiteralPath (Join-Path $tools "python.exe"))) {
        New-Item -ItemType Directory -Force $cache | Out-Null
        $package = Join-Path $cache "python.$PythonVersion.nupkg"
        $url = "https://api.nuget.org/v3-flatcontainer/python/$PythonVersion/python.$PythonVersion.nupkg"
        Write-Host "Downloading CPython $PythonVersion from $url"
        Invoke-WebRequest $url -OutFile $package -UseBasicParsing
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [IO.Compression.ZipFile]::ExtractToDirectory($package, (Join-Path $cache "package"))
        Move-Item (Join-Path $cache "package\tools") $tools
    }
    return $tools
}

# Version from pyproject.toml; Windows file versions must be numeric.
$version = (Select-String -LiteralPath (Join-Path $Root "pyproject.toml") -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
$numeric = (([regex]::Matches($version, '\d+') | Select-Object -First 4 | ForEach-Object Value) + @('0', '0', '0', '0'))[0..3] -join '.'
$iscc = Find-InnoCompiler
$pythonSource = Get-PythonRuntime
Write-Host "GWASMind $version (file version $numeric); Python runtime: $pythonSource"

if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Force $Stage | Out-Null

# 1. Private Python runtime without its test suite, IDLE and Tk.
$runtime = Join-Path $Stage "runtime"
robocopy $pythonSource $runtime /E /NFL /NDL /NJH /NJS /NP `
    /XD __pycache__ test idlelib tkinter turtledemo tcl /XF *.pyc | Out-Null
if ($LASTEXITCODE -ge 8) { throw "Copying the Python runtime failed (robocopy $LASTEXITCODE)." }
$python = Join-Path $runtime "python.exe"

# 2. Runtime dependencies: the "all" extra from pyproject.toml, nothing for development.
$requirements = & $python -c @"
import tomllib, pathlib
project = tomllib.loads(pathlib.Path(r'$Root\pyproject.toml').read_text(encoding='utf-8'))['project']
print('\n'.join(project['dependencies'] + project['optional-dependencies']['all']))
"@
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
Invoke-Checked $python (@("-m", "pip", "install", "--isolated", "--no-warn-script-location", "--no-compile") + @($requirements))
# Record the exact dependency versions shipped in this installer.
& $python -m pip freeze --isolated --all | Out-File (Join-Path $Stage "runtime-requirements.txt") -Encoding ascii

# 3. GWASMind source tree as in a checkout, so supervised analysis workers bind
#    to it exactly as with the launcher installation. In a git checkout, the
#    committed files of HEAD are extracted byte for byte, so local edits or a
#    line-ending conversion of the working tree never reach an installer.
#    The configuration freeze list (gwasmind.config_freeze.FREEZE_MANIFEST_NAME)
#    ships too, so installed copies verify their catalogs at start-up.
$freezeList = "docs/CONFIG_FREEZE_2026-09-28.json"
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git -and (Test-Path -LiteralPath (Join-Path $Root ".git"))) {
    $archive = Join-Path $Build "source.tar"
    Invoke-Checked $git.Source @("-C", $Root, "archive", "--format=tar", "-o", $archive, "HEAD", "src", "examples", "README.md", "LICENSE", $freezeList)
    Invoke-Checked "$env:WINDIR\System32\tar.exe" @("-xf", $archive, "-C", $Stage)
    Write-Host ("Source: commit {0}" -f (& $git.Source -C $Root rev-parse --short HEAD))
} else {
    robocopy (Join-Path $Root "src") (Join-Path $Stage "src") /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ *.egg-info | Out-Null
    robocopy (Join-Path $Root "examples") (Join-Path $Stage "examples") /E /NFL /NDL /NJH /NJS /NP | Out-Null
    Copy-Item -LiteralPath (Join-Path $Root "README.md"), (Join-Path $Root "LICENSE") -Destination $Stage
    New-Item -ItemType Directory -Force (Join-Path $Stage "docs") | Out-Null
    Copy-Item -LiteralPath (Join-Path $Root $freezeList) -Destination (Join-Path $Stage $freezeList)
}
# Relative path from runtime\Lib\site-packages to the source tree, so the
# installation works from any folder.
Set-Content -LiteralPath (Join-Path $runtime "Lib\site-packages\gwasmind-source.pth") -Value "../../../src" -Encoding ascii
Invoke-Checked $python @("-c", "import gwasmind, gwasmind.web.server, gwasmind.cli, duckdb, mcp, openai; print('runtime imports ok', gwasmind.__version__)")
Invoke-Checked $python @("-m", "compileall", "-q", "-j", "0", (Join-Path $runtime "Lib"), (Join-Path $Stage "src"))

# 4. Native launcher with the application icon.
$csc = Get-ChildItem "$env:WINDIR\Microsoft.NET\Framework64\v4*\csc.exe" | Select-Object -Last 1
if (-not $csc) { throw "The .NET Framework C# compiler (csc.exe) was not found." }
Invoke-Checked $csc.FullName @("/nologo", "/target:exe", "/optimize+", "/platform:x64",
    "/win32icon:$(Join-Path $Packaging 'gwasmind.ico')", "/out:$(Join-Path $Stage 'GWASMind.exe')",
    (Join-Path $Packaging "launcher\GWASMindLauncher.cs"))

# 5. Installer.
New-Item -ItemType Directory -Force $OutputDir | Out-Null
Invoke-Checked $iscc @("/Q", "/DAppVersion=$version", "/DNumericVersion=$numeric", "/DStageDir=$Stage",
    "/DOutputDir=$OutputDir", "/DSourceRoot=$Root", (Join-Path $Packaging "gwasmind.iss"))
$setup = Join-Path $OutputDir "GWASMind-$version-Setup.exe"
$hash = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLower()
"$hash  GWASMind-$version-Setup.exe" | Out-File "$setup.sha256" -Encoding ascii
Write-Host ("Built {0} ({1:N1} MB), SHA-256 {2}" -f $setup, ((Get-Item -LiteralPath $setup).Length / 1MB), $hash)
