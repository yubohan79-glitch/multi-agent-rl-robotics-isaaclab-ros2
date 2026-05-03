param(
    [string]$IsaacLabBat = "C:\Users\Administrator\IsaacLab\isaaclab.bat",
    [string]$PythonExe = "C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe",
    [string]$ScriptPath = "",
    [double]$Duration = 120,
    [switch]$Headless,
    [switch]$DemoFlow,
    [string]$RecordVideo = "",
    [ValidateSet("overview", "yellow_pov", "blue_pov")]
    [string]$RecordView = "overview",
    [int]$RecordFps = 24,
    [int]$RecordWidth = 1600,
    [int]$RecordHeight = 900,
    [switch]$DryRun,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($ScriptPath)) {
    $ScriptPath = Join-Path $RepoRoot "isaaclab_sim\robocup_visionrl_arena_sim.py"
}

$RuntimeRoot = Join-Path $RepoRoot ".isaaclab_runtime"
$DataDir = Join-Path $RuntimeRoot "data"
$LogDir = Join-Path $RuntimeRoot "logs"
$CacheDir = Join-Path $RuntimeRoot "cache"
$TempDir = Join-Path $RuntimeRoot "tmp"
$PipEnvDir = Join-Path $DataDir "pip3-envs\default"
$ExtCacheDir = Join-Path $DataDir "exts"

foreach ($dir in @($RuntimeRoot, $DataDir, $LogDir, $CacheDir, $TempDir, $PipEnvDir, $ExtCacheDir)) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

function Convert-ToKitPath([string]$PathValue) {
    return ($PathValue -replace "\\", "/")
}

$UserConfigPath = Convert-ToKitPath (Join-Path $DataDir "user.config.json")
$LogFile = Convert-ToKitPath (Join-Path $LogDir "kit_project.log")
$DumpDir = Convert-ToKitPath $DataDir
$PipEnv = Convert-ToKitPath $PipEnvDir
$ExtCache = Convert-ToKitPath $ExtCacheDir

$env:TMP = $TempDir
$env:TEMP = $TempDir
$env:OV_CACHE_PATH = $CacheDir
$env:OV_DATA_PATH = $DataDir
$env:OV_LOG_PATH = $LogDir
$env:OMNI_CACHE_PATH = $CacheDir
$env:OMNI_DATA_PATH = $DataDir
$env:OMNI_LOG_PATH = $LogDir

$KitArgs = @(
    "--/app/settings/persistent=false",
    "--/app/userConfigPath", $UserConfigPath,
    "--/log/file", $LogFile,
    "--/app/exts/registryCache", $ExtCache,
    "--/exts/omni.kit.pipapi/envPath", $PipEnv,
    "--/crashreporter/dumpDir", $DumpDir
) -join " "

$PythonArgs = @($ScriptPath, "--duration", "$Duration", "--kit_args", $KitArgs)
if ($Headless) {
    $PythonArgs += "--headless"
}
if ($DemoFlow) {
    $PythonArgs += "--demo_flow"
}
if (-not [string]::IsNullOrWhiteSpace($RecordVideo)) {
    $PythonArgs += @(
        "--record_video", $RecordVideo,
        "--record_view", $RecordView,
        "--record_fps", "$RecordFps",
        "--record_width", "$RecordWidth",
        "--record_height", "$RecordHeight"
    )
}
if ($ExtraArgs.Count -gt 0) {
    $PythonArgs += $ExtraArgs
}

Write-Host "[RoboCupVisionRL] Runtime root: $RuntimeRoot"
Write-Host "[RoboCupVisionRL] Kit user config: $UserConfigPath"
Write-Host "[RoboCupVisionRL] Launching IsaacLab with project-local cache/config."
if ($DryRun) {
    Write-Host "[RoboCupVisionRL] Dry run only. Command:"
    if (Test-Path $PythonExe) {
        Write-Host "$PythonExe $($PythonArgs -join ' ')"
    }
    else {
        Write-Host "$IsaacLabBat -p $($PythonArgs -join ' ')"
    }
    exit 0
}
if (Test-Path $PythonExe) {
    & $PythonExe @PythonArgs
}
else {
    Write-Warning "PythonExe not found; falling back to isaaclab.bat. Complex --kit_args may be less reliable through cmd.exe."
    & $IsaacLabBat @(@("-p") + $PythonArgs)
}
exit $LASTEXITCODE
