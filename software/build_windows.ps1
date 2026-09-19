$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

python -m pip install -r software/requirements.txt
python -m pip install -e . --no-deps
python -m PyInstaller --noconfirm --clean software/AI_Phenotype_Tool.spec

Write-Host "Built application: dist/AI_Phenotype_Tool/AI_Phenotype_Tool.exe"
