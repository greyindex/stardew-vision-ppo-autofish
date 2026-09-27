$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --python 3.12 .venv
    } else {
        py -3 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment.' }
}
$requirementFile = if (Test-Path -LiteralPath 'requirements.lock.txt') { 'requirements.lock.txt' } else { 'requirements.in' }
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python .venv\Scripts\python.exe --index https://download.pytorch.org/whl/cpu --index https://pypi.org/simple --index-strategy unsafe-best-match -r $requirementFile
} else {
    & .\.venv\Scripts\python.exe -m ensurepip
    & .\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r $requirementFile
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Training environment ready.'
