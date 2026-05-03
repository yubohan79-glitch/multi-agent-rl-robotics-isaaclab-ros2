param(
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"
$ScriptNeedle = "robocup_visionrl_arena_sim.py"
$AllowedNames = @("python.exe", "pythonw.exe", "kit.exe", "isaac-sim.exe")

try {
    $Processes = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and
        $_.CommandLine -and
        ($AllowedNames -contains $_.Name.ToLowerInvariant()) -and
        ($_.CommandLine -like "*RoboCupVisionRL_IsaacLab_ROS2*" -or $_.CommandLine -like "*$ScriptNeedle*")
    }
}
catch {
    Write-Error "Cannot inspect process command lines. Run this script from an elevated PowerShell window, or do not stop anything."
    exit 2
}

if (-not $Processes) {
    Write-Host "No RoboCupVisionRL IsaacLab processes found."
    exit 0
}

$Processes | Select-Object ProcessId, Name, CommandLine | Format-List

if ($WhatIfOnly) {
    Write-Host "WhatIfOnly set; no processes were stopped."
    exit 0
}

foreach ($Process in $Processes) {
    Write-Host "Stopping project process PID=$($Process.ProcessId)"
    Stop-Process -Id $Process.ProcessId -Force
}
