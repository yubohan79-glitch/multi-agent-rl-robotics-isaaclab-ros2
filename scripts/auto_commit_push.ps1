param(
    [string]$Repo = "",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$timestamp] $Message" | Tee-Object -FilePath $script:LogPath -Append
}

if (-not $Repo) {
    $Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
}

Set-Location -LiteralPath $Repo
$gitDir = Join-Path $Repo ".git"
$script:LogPath = Join-Path $gitDir "auto_commit.log"
$lockPath = Join-Path $gitDir "auto_commit.lock"

if (Test-Path -LiteralPath $lockPath) {
    $lockAge = (Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime
    if ($lockAge.TotalHours -lt 6) {
        Write-Log "Skip: another auto-commit run appears active."
        exit 0
    }
    Remove-Item -LiteralPath $lockPath -Force
}

New-Item -ItemType File -Path $lockPath -Force | Out-Null

try {
    $status = git status --porcelain
    if (-not $status) {
        Write-Log "No changes to commit."
        exit 0
    }

    git add -A
    $staged = git diff --cached --name-only
    if (-not $staged) {
        Write-Log "No staged changes after git add."
        exit 0
    }

    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Auto update: $stamp"
    git pull --rebase --autostash origin $Branch
    git push origin $Branch
    Write-Log "Committed and pushed auto update to origin/$Branch."
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)"
    exit 1
}
finally {
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Force
    }
}
