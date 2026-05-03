# setup.ps1 — TradeVision one-shot environment setup for Windows (PowerShell)
#
# Creates a project-local .venv\ and installs all dependencies into it.
# Never touches your global Python installation.
#
# Usage:
#   .\setup.ps1
#
# If you get an execution policy error, run first:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

#Requires -Version 5.1

$ErrorActionPreference = "Stop"

$VenvDir      = ".venv"
$Requirements = "requirements.txt"

function Write-Info    { param($msg) Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[setup] $msg" -ForegroundColor Green }
function Write-Warn    { param($msg) Write-Host "[setup] $msg" -ForegroundColor Yellow }
function Write-Err     { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ── Find Python 3.10+ ──────────────────────────────────────────────────────────
$PythonBin = $null
foreach ($candidate in @("python", "python3", "py")) {
    try {
        $ver = & $candidate --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                $PythonBin = $candidate
                Write-Info "Using Python: $ver"
                break
            }
        }
    } catch { continue }
}

if (-not $PythonBin) {
    Write-Err "Python 3.10+ not found in PATH.`nInstall from https://www.python.org/downloads/`nEnsure 'Add Python to PATH' is checked during install."
}

# ── Create virtual environment ─────────────────────────────────────────────────
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (Test-Path $VenvPython) {
    Write-Warn "Virtual environment already exists at $VenvDir\ -- skipping creation."
    Write-Warn "To rebuild: Remove-Item -Recurse -Force $VenvDir; .\setup.ps1"
} else {
    Write-Info "Creating virtual environment at .\$VenvDir\ ..."
    & $PythonBin -m venv $VenvDir
    Write-Success "Virtual environment created."
}

# ── Upgrade pip ────────────────────────────────────────────────────────────────
Write-Info "Upgrading pip ..."
& "$VenvDir\Scripts\python.exe" -m pip install --upgrade pip --quiet

# ── Install dependencies ───────────────────────────────────────────────────────
if (-not (Test-Path $Requirements)) {
    Write-Err "$Requirements not found. Are you in the project root?"
}

Write-Info "Installing dependencies from $Requirements ..."
& "$VenvDir\Scripts\pip.exe" install -r $Requirements
Write-Success "All dependencies installed."

# ── Print next steps ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║          TradeVision setup complete! ✅              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the venv before each session:" -ForegroundColor Cyan
Write-Host "    .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "    1. streamlit run app.py"
Write-Host "    2. Optional: python -m tradevision.model.trainer"
Write-Host "    3. Optional rebuild workflow: see README.md"
Write-Host ""
Write-Host "  Tip: if you see an execution policy error next time, run:" -ForegroundColor Yellow
Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
Write-Host ""
