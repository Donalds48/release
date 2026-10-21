# Einmalige Einrichtung: holt uv, damit ein eigenes Python 3.12 und alle Pakete in diesen Ordner.
# Nichts wird systemweit installiert – Ordner löschen = alles weg.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-Location $PSScriptRoot
Get-ChildItem -File | Unblock-File

$tools = Join-Path $PSScriptRoot '.tools'
$uv = Join-Path $tools 'uv.exe'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $tools 'python'
$env:UV_LINK_MODE = 'copy'

if (-not (Test-Path $uv)) {
    Write-Host 'Lade uv (Paketmanager) ...'
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = Join-Path $tools 'uv.zip'
    Invoke-WebRequest 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip' -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $tools -Force
    Remove-Item $zip
}

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Write-Host 'Richte Python 3.12 ein ...'
    & $uv venv .venv --python 3.12 --managed-python
    if ($LASTEXITCODE) { throw 'Python-Einrichtung fehlgeschlagen' }
}

Write-Host 'Installiere Pakete (beim ersten Mal ein paar Minuten) ...'
& $uv pip install --python $py -r requirements.txt
if ($LASTEXITCODE) { throw 'Paket-Installation fehlgeschlagen' }

# CUDA-Bibliotheken (~1 GB) nur mit NVIDIA-Grafikkarte, sonst läuft Whisper auf der CPU
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host 'NVIDIA-Grafikkarte gefunden - installiere CUDA-Bibliotheken ...'
    & $uv pip install --python $py -r requirements-gpu.txt
    if ($LASTEXITCODE) { Write-Host 'CUDA-Installation fehlgeschlagen - nutze CPU' }
} else {
    Write-Host 'Keine NVIDIA-Grafikkarte - Erkennung laeuft auf der CPU'
}

Set-Content (Join-Path $PSScriptRoot '.venv\.installed') 'ok'
Write-Host 'Einrichtung fertig.'
