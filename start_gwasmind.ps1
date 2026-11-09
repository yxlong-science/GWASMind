[CmdletBinding()]
param(
    [ValidateSet(
        "WebUI",
        "FrozenWebUI",
        "Menu",
        "UnitTests",
        "LiveAnalysis",
        "FrozenCapabilities",
        "FrozenAnalysis",
        "LiveServer",
        "FrozenServer",
        "Help"
    )]
    [string]$Mode = "WebUI",
    [string]$InputPath = "",
    [string]$OutputRoot = "",
    [string]$ListenHost = "",
    [Nullable[int]]$Port = $null,
    [Nullable[int]]$DuckDBThreads = $null,
    [string]$DuckDBMemoryLimit = "",
    [string]$FrozenDataRoot = "",
    [switch]$SkipInstall,
    [switch]$NoPause
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script:VenvRoot = Join-Path $script:ProjectRoot ".venv"
$script:VenvPython = Join-Path $script:VenvRoot "Scripts\python.exe"
# Explicit option, then GWASMIND_OT_DATA_ROOT (see .env.example), then data\opentargets.
$script:FrozenDataRoot = if ($FrozenDataRoot) { $FrozenDataRoot }
    elseif ($env:GWASMIND_OT_DATA_ROOT) { $env:GWASMIND_OT_DATA_ROOT }
    else { Join-Path $script:ProjectRoot "data\opentargets" }
if (-not [IO.Path]::IsPathRooted($script:FrozenDataRoot)) {
    $script:FrozenDataRoot = Join-Path $script:ProjectRoot $script:FrozenDataRoot
}
$script:PythonPath = ""
$script:RuntimeParameters = $null
$script:LauncherSettings = $null

$env:PYTHONUTF8 = "1"
if ([string]::IsNullOrWhiteSpace($env:GWASMIND_LDLINK_CREDENTIAL_FILE)) {
    $env:GWASMIND_LDLINK_CREDENTIAL_FILE = Join-Path $script:ProjectRoot ".gwasmind\credentials\ldlink.dpapi"
}

function Get-LauncherParameter {
    param([Parameter(Mandatory = $true)][string]$Key)

    $entry = $script:RuntimeParameters.PSObject.Properties[$Key]
    if ($null -eq $entry -or $null -eq $entry.Value) {
        throw "The runtime parameter catalog does not define '$Key'."
    }
    return $entry.Value
}

function Initialize-LauncherConfiguration {
    if ($null -ne $script:LauncherSettings) {
        return
    }
    # Use the canonical Python loader and its validation, never a second TOML parser.
    # Calling main avoids runpy's duplicate-import warning from package initialization.
    $rawParameters = & $script:PythonPath -B -c (
        "from gwasmind.configuration import main; main()"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime parameter validation failed before startup (exit code $LASTEXITCODE)."
    }
    $script:RuntimeParameters = ($rawParameters -join "`n") | ConvertFrom-Json

    $threadCount = if ($null -ne $DuckDBThreads) {
        $DuckDBThreads
    }
    elseif (-not [string]::IsNullOrWhiteSpace($env:DUCKDB_THREADS)) {
        $env:DUCKDB_THREADS
    }
    else {
        Get-LauncherParameter "execution.duckdb_threads"
    }
    $parsedThreads = 0
    if (-not [int]::TryParse([string]$threadCount, [ref]$parsedThreads) -or
        $parsedThreads -lt 1 -or $parsedThreads -gt 8) {
        throw "DuckDB thread count must be an integer between 1 and 8."
    }
    $memoryLimit = if ($DuckDBMemoryLimit) {
        $DuckDBMemoryLimit
    }
    elseif ($env:GWASMIND_DUCKDB_MEMORY_LIMIT) {
        $env:GWASMIND_DUCKDB_MEMORY_LIMIT
    }
    else {
        Get-LauncherParameter "execution.duckdb_memory_limit"
    }
    if ([string]$memoryLimit -notmatch '^[1-9][0-9]*(MB|GB)$') {
        throw "DuckDB memory limit must be a positive MB or GB value."
    }
    $bindAddress = if ($ListenHost) { $ListenHost }
        elseif ($env:GWASMIND_HOST) { $env:GWASMIND_HOST }
        else { Get-LauncherParameter "web.host" }
    if ([string]::IsNullOrWhiteSpace($bindAddress)) {
        throw "A server bind address is required."
    }
    $portNumber = if ($null -ne $Port) { $Port }
        elseif ($env:GWASMIND_PORT) { $env:GWASMIND_PORT }
        else { Get-LauncherParameter "web.launcher_port" }
    $parsedPort = 0
    if (-not [int]::TryParse([string]$portNumber, [ref]$parsedPort) -or
        $parsedPort -lt 0 -or $parsedPort -gt 65535) {
        throw "The server port must be an integer between 0 and 65535."
    }

    foreach ($settingName in @(
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
    )) {
        $current = [Environment]::GetEnvironmentVariable($settingName, [EnvironmentVariableTarget]::Process)
        if ([string]::IsNullOrWhiteSpace($current)) {
            $current = [string](Get-LauncherParameter "execution.duckdb_threads")
        }
        $numericThreads = 0
        if (-not [int]::TryParse($current, [ref]$numericThreads) -or $numericThreads -lt 1) {
            throw "$settingName must be a positive integer."
        }
        [Environment]::SetEnvironmentVariable($settingName, $current, [EnvironmentVariableTarget]::Process)
    }
    $env:DUCKDB_THREADS = [string]$parsedThreads
    $script:LauncherSettings = [PSCustomObject]@{
        BindAddress = [string]$bindAddress
        Port = $parsedPort
        DuckDBThreads = $parsedThreads
        DuckDBMemoryLimit = ([string]$memoryLimit).ToUpperInvariant()
    }
}

function Test-DeepSeekEndpoint {
    param([string]$BaseUrl)

    $parsed = $null
    if (-not [Uri]::TryCreate($BaseUrl, [UriKind]::Absolute, [ref]$parsed)) {
        return $false
    }
    $name = $parsed.DnsSafeHost.TrimEnd('.').ToLowerInvariant()
    return $parsed.Scheme -eq "https" -and (
        $name -eq "api.deepseek.com" -or $name.EndsWith(".deepseek.com")
    )
}

function Write-Banner {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " GWASMind launcher" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " Project : $script:ProjectRoot"
    Write-Host " Live primary : official Open Targets GraphQL API (optional hosted MCP; no download)"
    Write-Host " Coverage gap : one bounded Ensembl GRCh38 region lookup only when all exact OT variants are absent"
    Write-Host " Frozen  : manifest-verified local snapshot; resources from shared configuration"
    Write-Host ""
}

function Show-Help {
    Write-Banner
    Write-Host "Double-click start_gwasmind.bat to launch the graphical interface."
    Write-Host ""
    Write-Host "Automation examples:"
    Write-Host "  .\start_gwasmind.bat -Mode WebUI -NoPause"
    Write-Host "  .\start_gwasmind.bat -Mode FrozenWebUI -NoPause"
    Write-Host "  .\start_gwasmind.bat -Mode Menu"
    Write-Host "  .\start_gwasmind.bat -Mode UnitTests -NoPause"
    Write-Host "  .\start_gwasmind.bat -Mode LiveAnalysis -InputPath examples\locus.live.asthma.json -NoPause"
    Write-Host "  .\start_gwasmind.bat -Mode FrozenCapabilities -NoPause"
    Write-Host ""
    Write-Host "Frozen modes require gwasmind_snapshot_manifest.json in the data root."
    Write-Host "The data root is -FrozenDataRoot <path>, else GWASMIND_OT_DATA_ROOT, else data\opentargets."
    Write-Host "Provision it explicitly with:"
    Write-Host "  gwasmind inspect-snapshot --data-root <root> --release <release> --output <root>\gwasmind_snapshot_manifest.json"
    Write-Host ""
    Write-Host "The first normal run creates .venv and installs .[all,dev]."
    Write-Host "Use -SkipInstall only when running with an already prepared Python environment."
    Write-Host "Runtime defaults come from the parameter catalog (GWASMIND_PARAMETERS_FILE may select overrides)."
    Write-Host "Optional overrides: -ListenHost <address> -Port <port> -DuckDBThreads <count> -DuckDBMemoryLimit <size>."
    Write-Host "Explicit options take precedence over GWASMIND_HOST/GWASMIND_PORT/DUCKDB_THREADS/GWASMIND_DUCKDB_MEMORY_LIMIT."
    Write-Host "Existing OMP/OPENBLAS/MKL/NUMEXPR thread settings are preserved."
    Write-Host "DuckDB options apply to frozen CLI modes; GUI worker resource limits use the shared parameter file."
}

function Show-Menu {
    Write-Banner
    Write-Host "  1  Launch the GWASMind graphical interface"
    Write-Host "  2  Run unit and contract tests (no network, no database scan)"
    Write-Host "  3  Run a live sample analysis (latest official data)"
    Write-Host "  4  Inspect frozen capabilities (does not open Parquet files)"
    Write-Host "  5  Run a frozen sample analysis (verified local snapshot)"
    Write-Host "  6  Start live semantic MCP server over stdio (advanced)"
    Write-Host "  7  Start frozen semantic MCP server over stdio (advanced)"
    Write-Host "  8  Launch the graphical interface with local frozen evidence"
    Write-Host "  0  Exit"
    Write-Host ""
}

function Get-SystemPython {
    # The Microsoft Store alias (WindowsApps\python.exe) is on PATH even when
    # Python is not installed, so every candidate must pass the version check.
    foreach ($name in @("py", "python", "python3")) {
        foreach ($command in @(Get-Command $name -All -ErrorAction SilentlyContinue)) {
            if (Test-PythonExecutable -Path $command.Source) {
                return $command.Source
            }
        }
    }
    throw (
        "Python 3.11 or newer was not found. Install it from " +
        "https://www.python.org/downloads/ and run this launcher again."
    )
}

function Test-PythonExecutable {
    param([Parameter(Mandatory = $true)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    try {
        & $Path -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-LongPathSupport {
    # Saved analyses use nested, hash-named folders; without Windows long-path
    # support, saving large results can fail once paths exceed 260 characters.
    $setting = Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name LongPathsEnabled -ErrorAction SilentlyContinue
    if ($null -ne $setting -and $setting.LongPathsEnabled -eq 1) {
        return
    }
    Write-Warning (
        "Windows long-path support is disabled. Saving very large results may fail because " +
        "their file paths can exceed 260 characters. Install GWASMind in a short folder such as " +
        "C:\GWASMind, and ask an administrator to enable long paths " +
        "(https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation)."
    )
}

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)

    & $script:PythonPath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Initialize-GWASMind {
    if ($script:PythonPath) {
        Initialize-LauncherConfiguration
        return
    }

    if ($SkipInstall) {
        if (Test-PythonExecutable -Path $script:VenvPython) {
            $script:PythonPath = $script:VenvPython
        }
        else {
            $script:PythonPath = Get-SystemPython
        }
        $sourceRoot = Join-Path $script:ProjectRoot "src"
        if ($env:PYTHONPATH) {
            $env:PYTHONPATH = "$sourceRoot$([IO.Path]::PathSeparator)$env:PYTHONPATH"
        }
        else {
            $env:PYTHONPATH = $sourceRoot
        }
    }
    else {
        if (-not (Test-PythonExecutable -Path $script:VenvPython)) {
            $systemPython = Get-SystemPython
            $venvArguments = @("-m", "venv")
            if (Test-Path -LiteralPath $script:VenvRoot -PathType Container) {
                Write-Host "Repairing the stale project virtual environment..." -ForegroundColor Yellow
                $venvArguments += "--clear"
            }
            else {
                Write-Host "Creating project virtual environment..." -ForegroundColor Yellow
            }
            $venvArguments += $script:VenvRoot
            if ((Split-Path -Leaf $systemPython) -ieq "py.exe") {
                & $systemPython -3 @venvArguments
            }
            else {
                & $systemPython @venvArguments
            }
            if ($LASTEXITCODE -ne 0) {
                throw "Could not create the project virtual environment."
            }
        }

        $script:PythonPath = $script:VenvPython
        $missing = & $script:PythonPath -c (
            "import importlib.util, pathlib, sys; " +
            "root=pathlib.Path(sys.argv[1]).resolve(); " +
            "names=('gwasmind','duckdb','mcp','openai','pytest'); " +
            "missing=[name for name in names if importlib.util.find_spec(name) is None]; " +
            "spec=importlib.util.find_spec('gwasmind'); " +
            "origin=pathlib.Path(spec.origin).resolve() if spec and spec.origin else None; " +
            "missing += ['gwasmind-current-project'] if origin and root not in origin.parents else []; " +
            "print(','.join(missing)); raise SystemExit(bool(missing))"
        ) $script:ProjectRoot
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing GWASMind and test dependencies..." -ForegroundColor Yellow
            if ($missing) {
                Write-Host "Missing: $missing"
            }
            Push-Location $script:ProjectRoot
            try {
                Invoke-CheckedPython -ArgumentList @(
                    "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-input", "-e", ".[all,dev]"
                )
            }
            finally {
                Pop-Location
            }
        }
    }

    & $script:PythonPath -c (
        "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "GWASMind requires Python 3.11 or newer."
    }
    Initialize-LauncherConfiguration
    Test-LongPathSupport
    Write-Host "Python  : $script:PythonPath" -ForegroundColor DarkGray
    Write-Host "DuckDB  : $($script:LauncherSettings.DuckDBThreads) threads, $($script:LauncherSettings.DuckDBMemoryLimit) memory" -ForegroundColor DarkGray
}

function Resolve-AnalysisInput {
    param([string]$DefaultInput = "examples\locus.frozen.positive.json")
    $selected = $InputPath
    if (-not $selected -and $Mode -eq "Menu") {
        $selected = Read-Host (
            "Input locus JSON [$DefaultInput]"
        )
    }
    if (-not $selected) {
        $selected = $DefaultInput
    }
    if (-not [IO.Path]::IsPathRooted($selected)) {
        $selected = Join-Path $script:ProjectRoot $selected
    }
    if (-not (Test-Path -LiteralPath $selected -PathType Leaf)) {
        throw "Input JSON does not exist: $selected"
    }
    return (Resolve-Path -LiteralPath $selected).Path
}

function New-TestOutputDirectory {
    param([Parameter(Mandatory = $true)][string]$Name)

    $base = $OutputRoot
    if (-not $base) {
        $base = Join-Path $script:ProjectRoot "artifacts\manual_tests"
    }
    elseif (-not [IO.Path]::IsPathRooted($base)) {
        $base = Join-Path $script:ProjectRoot $base
    }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $path = Join-Path $base "${Name}_$stamp"
    New-Item -ItemType Directory -Force -Path $path | Out-Null
    return (Resolve-Path -LiteralPath $path).Path
}

function Show-AnalysisExitCode {
    param([Parameter(Mandatory = $true)][int]$ExitCode)

    switch ($ExitCode) {
        0 { Write-Host "Analysis completed with a prioritized result or reviewed evidence." -ForegroundColor Green }
        2 { Write-Host "Analysis stopped safely: invalid input or no significant GWAS signal." -ForegroundColor Yellow }
        3 { Write-Host "Analysis completed without prioritization because evidence was insufficient." -ForegroundColor Yellow }
        4 { throw "The selected evidence service was unavailable." }
        default { throw "Analysis failed with exit code $ExitCode." }
    }
}

function Invoke-WebUI {
    param(
        [ValidateSet("live", "frozen-safe")]
        [string]$AnalysisMode = "live"
    )

    if ($AnalysisMode -eq "frozen-safe" -and -not (
        Test-Path -LiteralPath $script:FrozenDataRoot -PathType Container
    )) {
        throw "Frozen graphical mode requires: $script:FrozenDataRoot"
    }
    if ($AnalysisMode -eq "frozen-safe" -and -not (
        Test-Path -LiteralPath (
            Join-Path $script:FrozenDataRoot "gwasmind_snapshot_manifest.json"
        ) -PathType Leaf
    )) {
        throw (
            "Frozen graphical mode requires a verified " +
            "gwasmind_snapshot_manifest.json in: $script:FrozenDataRoot"
        )
    }
    Write-Host "Starting the local GWASMind graphical interface ($AnalysisMode)..." -ForegroundColor Yellow
    Write-Host "Bind address: $($script:LauncherSettings.BindAddress); port: $($script:LauncherSettings.Port) (0 selects an available port)."
    Write-Host "Close this window or press Ctrl+C to stop the local service."
    $arguments = @(
        "-m", "gwasmind.web.server",
        "--host", $script:LauncherSettings.BindAddress,
        "--port", [string]$script:LauncherSettings.Port,
        "--default-mode", $AnalysisMode,
        "--open-browser"
    )
    if ($AnalysisMode -eq "frozen-safe") {
        $arguments += @("--frozen-data-root", $script:FrozenDataRoot)
    }
    $modelKeyEnvironment = [string]$env:GWASMIND_MODEL_API_KEY_ENV
    $modelBaseUrl = [string]$env:GWASMIND_MODEL_BASE_URL
    $modelId = [string]$env:GWASMIND_MODEL_ID
    $modelThinkingMode = [string]$env:GWASMIND_MODEL_THINKING_MODE
    $modelReasoningEffort = [string]$env:GWASMIND_MODEL_REASONING_EFFORT
    $customModelRequested = (
        $modelKeyEnvironment -or $modelBaseUrl -or $modelId
    )
    $enableEnvironmentModel = $false

    if ($customModelRequested) {
        if (-not ($modelKeyEnvironment -and $modelBaseUrl -and $modelId)) {
            Write-Warning (
                "Ignoring incomplete GWASMIND model configuration. Set " +
                "GWASMIND_MODEL_API_KEY_ENV, GWASMIND_MODEL_BASE_URL, and " +
                "GWASMIND_MODEL_ID together, or configure the model in Settings."
            )
        }
        else {
            $configuredModelKey = [Environment]::GetEnvironmentVariable(
                $modelKeyEnvironment,
                [EnvironmentVariableTarget]::Process
            )
            if ([string]::IsNullOrWhiteSpace($configuredModelKey)) {
                Write-Warning (
                    "Ignoring GWASMIND model configuration because environment " +
                    "variable '$modelKeyEnvironment' is empty. Use Settings instead."
                )
            }
            else {
                $enableEnvironmentModel = $true
            }
        }
    }
    elseif ($env:DEEPSEEK_API_KEY) {
        # A DeepSeek credential may only be paired automatically with its
        # official HTTPS endpoint, even if shared defaults select another provider.
        $modelKeyEnvironment = "DEEPSEEK_API_KEY"
        $modelBaseUrl = [string](Get-LauncherParameter "model.default_base_url")
        $modelId = [string](Get-LauncherParameter "model.default_id")
        $enableEnvironmentModel = Test-DeepSeekEndpoint -BaseUrl $modelBaseUrl
        if (-not $enableEnvironmentModel) {
            Write-Warning "The configured default endpoint is not DeepSeek. Configure its model credential in Settings."
        }
    }
    elseif ($env:OPENAI_API_KEY) {
        Write-Host (
            "OPENAI_API_KEY detected. Choose the intended OpenAI model in Settings; " +
            "GWASMind will not guess a model ID."
        ) -ForegroundColor Yellow
    }

    if ($enableEnvironmentModel) {
        if ([string]::IsNullOrWhiteSpace($modelThinkingMode)) {
            $modelThinkingMode = if (
                $modelId -eq [string](Get-LauncherParameter "model.default_id") -and
                (Test-DeepSeekEndpoint -BaseUrl $modelBaseUrl)
            ) {
                [string](Get-LauncherParameter "model.first_run_thinking_mode")
            }
            else {
                "disabled"
            }
        }
        if ($modelThinkingMode -notin @("enabled", "disabled")) {
            throw "GWASMIND_MODEL_THINKING_MODE must be 'enabled' or 'disabled'."
        }
        if ([string]::IsNullOrWhiteSpace($modelReasoningEffort)) {
            $modelReasoningEffort = [string](Get-LauncherParameter "model.default_reasoning_effort")
        }
        if ($modelReasoningEffort -notin @("low", "high", "max")) {
            throw "GWASMIND_MODEL_REASONING_EFFORT must be 'low', 'high', or 'max'."
        }
        $arguments += @(
            "--model-api-key-env", $modelKeyEnvironment,
            "--model-base-url", $modelBaseUrl,
            "--model-id", $modelId,
            "--model-thinking-mode", $modelThinkingMode,
            "--model-reasoning-effort", $modelReasoningEffort
        )
    }
    # server.main also reads these variables. An ignored configuration must
    # not be reactivated by its environment fallback in the child process.
    $ignoredModelSettings = @{}
    try {
        if (-not $enableEnvironmentModel) {
            foreach ($settingName in @(
                "GWASMIND_MODEL_API_KEY_ENV", "GWASMIND_MODEL_BASE_URL",
                "GWASMIND_MODEL_ID", "GWASMIND_MODEL_THINKING_MODE",
                "GWASMIND_MODEL_REASONING_EFFORT"
            )) {
                $ignoredModelSettings[$settingName] = [Environment]::GetEnvironmentVariable(
                    $settingName, [EnvironmentVariableTarget]::Process
                )
                [Environment]::SetEnvironmentVariable(
                    $settingName, $null, [EnvironmentVariableTarget]::Process
                )
            }
        }
        & $script:PythonPath @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "The graphical interface exited with code $LASTEXITCODE."
        }
    }
    finally {
        foreach ($settingName in $ignoredModelSettings.Keys) {
            [Environment]::SetEnvironmentVariable(
                $settingName, $ignoredModelSettings[$settingName],
                [EnvironmentVariableTarget]::Process
            )
        }
    }
}

function Invoke-UnitTests {
    # The public distribution ships without the development test suite.
    if (-not (Test-Path -LiteralPath (Join-Path $script:ProjectRoot "tests") -PathType Container)) {
        Write-Host "The test suite is not included in this distribution." -ForegroundColor Yellow
        return
    }
    Write-Host "Running sequential tests; no real database or live API is opened..." -ForegroundColor Yellow
    $previous = $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
    Push-Location $script:ProjectRoot
    try {
        Invoke-CheckedPython -ArgumentList @("-m", "pytest", "-q")
    }
    finally {
        Pop-Location
        if ($null -eq $previous) {
            Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = $previous
        }
    }

    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($null -eq $nodeCommand) {
        Write-Warning "Node.js was not found; browser JavaScript tests were not run."
        return
    }
    $uiTestFiles = @(
        Get-ChildItem -LiteralPath (Join-Path $script:ProjectRoot "tests\js") `
            -Filter "*.mjs" -File |
            Sort-Object FullName |
            ForEach-Object { $_.FullName }
    )
    if ($uiTestFiles.Count -eq 0) {
        throw "No browser JavaScript tests were found under tests\js."
    }
    Write-Host "Running browser JavaScript tests sequentially..." -ForegroundColor Yellow
    # The JS fixtures run the production Python parser through $env:PYTHON. A bare
    # "python" can be the Microsoft Store alias without an installed interpreter.
    $previousPython = $env:PYTHON
    $env:PYTHON = $script:PythonPath
    try {
        & $nodeCommand.Source --test @uiTestFiles
        $nodeExitCode = $LASTEXITCODE
    }
    finally {
        if ($null -eq $previousPython) {
            Remove-Item Env:PYTHON -ErrorAction SilentlyContinue
        }
        else {
            $env:PYTHON = $previousPython
        }
    }
    if ($nodeExitCode -ne 0) {
        throw "Browser JavaScript tests failed with code $nodeExitCode."
    }
}

function Invoke-LiveAnalysis {
    # Live gene queries need a recognized phenotype; the frozen fixtures use a placeholder trait.
    $inputFile = Resolve-AnalysisInput -DefaultInput "examples\locus.live.asthma.json"
    $output = New-TestOutputDirectory -Name "live"
    Write-Host "Querying the official live Open Targets GraphQL API as the primary data source..." -ForegroundColor Yellow
    Write-Host "If all exact OT variants are absent, eligible non-benchmark gene queries may make one bounded Ensembl GRCh38 region-overlap lookup."
    Write-Host "No Open Targets dataset will be downloaded."
    Write-Host "Evidence is retrieved and reviewed only; model ranking of live genes runs from the graphical interface."
    & $script:PythonPath -m gwasmind.cli analyze `
        --input $inputFile `
        --mode live `
        --ranker deterministic `
        --defer-ranking `
        --output-dir $output
    $code = $LASTEXITCODE
    Show-AnalysisExitCode -ExitCode $code
    Write-Host "Artifacts: $output" -ForegroundColor Cyan
}

function Invoke-FrozenCapabilities {
    if (-not (Test-Path -LiteralPath $script:FrozenDataRoot -PathType Container)) {
        throw "Frozen Open Targets directory does not exist: $script:FrozenDataRoot"
    }
    $output = New-TestOutputDirectory -Name "frozen_capabilities"
    $report = Join-Path $output "mcp_capabilities.json"
    Write-Host "Inspecting directories only; Parquet files will not be opened..." -ForegroundColor Yellow
    $arguments = @(
        "-m", "gwasmind.cli", "inspect-mcp-capabilities",
        "--data-root", $script:FrozenDataRoot,
        "--duckdb-threads", [string]$script:LauncherSettings.DuckDBThreads,
        "--duckdb-memory-limit", $script:LauncherSettings.DuckDBMemoryLimit,
        "--output", $report
    )
    & $script:PythonPath @arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen capability inspection failed with exit code $LASTEXITCODE."
    }
    Write-Host "Report: $report" -ForegroundColor Cyan
}

function Invoke-FrozenAnalysis {
    if (-not (Test-Path -LiteralPath $script:FrozenDataRoot -PathType Container)) {
        throw "Frozen Open Targets directory does not exist: $script:FrozenDataRoot"
    }
    $inputFile = Resolve-AnalysisInput
    $output = New-TestOutputDirectory -Name "frozen"
    Write-Host "Running against the manifest-verified local Open Targets snapshot..." -ForegroundColor Yellow
    Write-Host "DuckDB is limited to $($script:LauncherSettings.DuckDBThreads) threads and $($script:LauncherSettings.DuckDBMemoryLimit) memory."
    & $script:PythonPath -m gwasmind.cli analyze `
        --input $inputFile `
        --mode frozen-safe `
        --data-root $script:FrozenDataRoot `
        --duckdb-threads $script:LauncherSettings.DuckDBThreads `
        --duckdb-memory-limit $script:LauncherSettings.DuckDBMemoryLimit `
        --ranker deterministic `
        --offline `
        --output-dir $output
    $code = $LASTEXITCODE
    Show-AnalysisExitCode -ExitCode $code
    Write-Host "Artifacts: $output" -ForegroundColor Cyan
}

function Invoke-LiveServer {
    Write-Host "Starting the live stdio MCP server. Stop it with Ctrl+C." -ForegroundColor Yellow
    Write-Host "MCP clients should call .venv\Scripts\gwasmind.exe directly, not this menu."
    & $script:PythonPath -m gwasmind.cli serve-mcp --mode live
    if ($LASTEXITCODE -ne 0) {
        throw "Live MCP server exited with code $LASTEXITCODE."
    }
}

function Invoke-FrozenServer {
    if (-not (Test-Path -LiteralPath $script:FrozenDataRoot -PathType Container)) {
        throw "Frozen Open Targets directory does not exist: $script:FrozenDataRoot"
    }
    Write-Host "Starting the frozen stdio MCP server. Stop it with Ctrl+C." -ForegroundColor Yellow
    Write-Host "DuckDB is limited to $($script:LauncherSettings.DuckDBThreads) threads and $($script:LauncherSettings.DuckDBMemoryLimit) memory."
    & $script:PythonPath -m gwasmind.cli serve-mcp `
        --mode frozen-safe `
        --data-root $script:FrozenDataRoot `
        --duckdb-threads $script:LauncherSettings.DuckDBThreads `
        --duckdb-memory-limit $script:LauncherSettings.DuckDBMemoryLimit
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen MCP server exited with code $LASTEXITCODE."
    }
}

function Invoke-SelectedMode {
    param([Parameter(Mandatory = $true)][string]$SelectedMode)

    Initialize-GWASMind
    switch ($SelectedMode) {
        "WebUI" { Invoke-WebUI -AnalysisMode "live" }
        "FrozenWebUI" { Invoke-WebUI -AnalysisMode "frozen-safe" }
        "UnitTests" { Invoke-UnitTests }
        "LiveAnalysis" { Invoke-LiveAnalysis }
        "FrozenCapabilities" { Invoke-FrozenCapabilities }
        "FrozenAnalysis" { Invoke-FrozenAnalysis }
        "LiveServer" { Invoke-LiveServer }
        "FrozenServer" { Invoke-FrozenServer }
        default { throw "Unknown launcher mode: $SelectedMode" }
    }
}

function Invoke-InteractiveMenu {
    while ($true) {
        Show-Menu
        $choice = Read-Host "Select"
        $selected = switch ($choice) {
            "1" { "WebUI" }
            "2" { "UnitTests" }
            "3" { "LiveAnalysis" }
            "4" { "FrozenCapabilities" }
            "5" { "FrozenAnalysis" }
            "6" { "LiveServer" }
            "7" { "FrozenServer" }
            "8" { "FrozenWebUI" }
            "0" { return }
            default { "" }
        }
        if (-not $selected) {
            Write-Host "Unknown selection: $choice" -ForegroundColor Red
            Start-Sleep -Seconds 1
            continue
        }
        try {
            Invoke-SelectedMode -SelectedMode $selected
        }
        catch {
            Write-Host ""
            Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        }
        Write-Host ""
        [void](Read-Host "Press Enter to return to the menu")
    }
}

try {
    Set-Location -LiteralPath $script:ProjectRoot
    if ($Mode -eq "Help") {
        Show-Help
    }
    elseif ($Mode -eq "Menu") {
        Invoke-InteractiveMenu
    }
    else {
        Write-Banner
        Invoke-SelectedMode -SelectedMode $Mode
    }
    exit 0
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    if (-not $NoPause) {
        [void](Read-Host "Press Enter to close")
    }
    exit 1
}
