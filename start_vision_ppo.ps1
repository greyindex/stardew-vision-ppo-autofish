param(
    [switch]$CheckOnly,
    [switch]$InstallOnly,
    [switch]$ForceInstall
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $root 'training\.venv-vision\Scripts\python.exe'
$pythonw = Join-Path $root 'training\.venv-vision\Scripts\pythonw.exe'
$check = Join-Path $root 'runtime_check.py'
$requirements = Join-Path $root 'requirements-vision-gui-lock.txt'

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Failure)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Failure (exit $LASTEXITCODE)" }
}

try {
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host 'Creating the Python 3.12 vision environment...'
        $directory = Join-Path $root 'training\.venv-vision'
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Invoke-Checked { uv venv --python 3.12 $directory } 'Could not create Python environment'
        } elseif (Get-Command py -ErrorAction SilentlyContinue) {
            Invoke-Checked { py -3.12 -m venv $directory } 'Could not create Python environment; install uv or Python 3.12'
        } else {
            throw 'Install uv (recommended) or Python 3.12 before starting.'
        }
    }

    $needsInstall = $ForceInstall -or -not (Test-Path -LiteralPath $pythonw)
    if (-not $needsInstall) {
        & $python $check --quick
        $checkCode = $LASTEXITCODE
        if ($checkCode -eq 2) { $needsInstall = $true }
        elseif ($checkCode -ne 0) { throw "Runtime check failed (exit $checkCode). See the message above." }
    }

    if ($needsInstall) {
        Write-Host 'Installing pinned dependencies. The first run downloads the CUDA PyTorch wheel and may take several minutes...'
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            Invoke-Checked {
                uv pip install --python $python --index https://download.pytorch.org/whl/cu128 --index https://pypi.org/simple --index-strategy unsafe-best-match -r $requirements
            } 'Dependency installation failed'
        } else {
            & $python -m pip --version *> $null
            if ($LASTEXITCODE -ne 0) {
                Invoke-Checked { & $python -m ensurepip --upgrade } 'Could not install pip'
            }
            Invoke-Checked {
                & $python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r $requirements
            } 'Dependency installation failed'
        }
    }

    if ($CheckOnly) {
        Invoke-Checked { & $python $check --full } 'Full runtime check failed'
        Write-Host 'Vision + PPO runtime check passed.'
        exit 0
    }

    Invoke-Checked { & $python $check --quick } 'Runtime check failed'
    if ($InstallOnly) {
        Write-Host 'The environment is ready.'
        exit 0
    }

    $gui = Join-Path $root 'vision_ppo_gui.py'
    Write-Host 'Starting the Vision + PPO GUI...'
    Start-Process -FilePath $pythonw -ArgumentList ('"{0}"' -f $gui) -WorkingDirectory $root -WindowStyle Normal
    exit 0
} catch {
    Write-Error $_.Exception.Message
    exit 1
}
