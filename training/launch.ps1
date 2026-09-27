$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$trainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainingPython)) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
}
$available = $false
try {
    $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/status' -TimeoutSec 2
    $available = $null -ne $status.training
} catch { }
if (-not $available) {
    Start-Process -FilePath $trainingPython -ArgumentList @('-u','-m','fishing_sim.serve','--port','8767') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'server.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'server.stderr.log') | Out-Null
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 300
        try {
            $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8767/api/status' -TimeoutSec 1
            $available = $null -ne $status.training
            if ($available) { break }
        } catch { }
    }
}
if (-not $available) { throw 'Simulator could not start. See training/server.stderr.log.' }
Start-Process 'http://127.0.0.1:8767/'
