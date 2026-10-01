$ErrorActionPreference = 'Stop'
$receiptProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $receiptProjectRoot
$receiptPython = Join-Path $receiptProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $receiptPython)) {
    throw 'Create the local .venv and install requirements-dev.txt first; see README.md.'
}
& $receiptPython -m uvicorn receiptlab.api:app --host 127.0.0.1 --port 8010
