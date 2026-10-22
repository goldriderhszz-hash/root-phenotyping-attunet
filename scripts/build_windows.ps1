param(
    [string]$Python = "python",
    [switch]$SkipBuild,
    [string]$TclZlib = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repo
try {
    if (-not $SkipBuild) {
        $extraBinaries = @()
        if ($TclZlib) {
            $dependency = (Resolve-Path -LiteralPath $TclZlib).Path
            $extraBinaries = @("--add-binary", "$dependency;.")
        }
        & $Python scripts/freeze_build_info.py
        if ($LASTEXITCODE -ne 0) { throw "Source fingerprint failed" }
        & $Python -m PyInstaller --noconfirm --clean --windowed --onedir `
            --name RootScope --collect-all tkinterdnd2 `
            --exclude-module torch --exclude-module torchvision --exclude-module torchaudio `
            --exclude-module sklearn --exclude-module joblib --exclude-module matplotlib --exclude-module pandas `
            --add-data "model;model" --add-data "rootscope/translations.json;rootscope" `
            --add-data "rootscope/build_info.json;rootscope" --add-data "templates;templates" @extraBinaries desktop.py
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path (Join-Path $repo "dist\RootScope\RootScope.exe"))) {
        throw "RootScope.exe not found; build before packaging"
    }
    & $Python scripts/package_runtime_docs.py
    if ($LASTEXITCODE -ne 0) { throw "Packaging guides/notices failed" }

    $release = Join-Path $repo "release"
    New-Item -ItemType Directory -Path $release -Force | Out-Null
    $archive = Join-Path $release "RootScope-Windows-x64.zip"
    Compress-Archive -Path (Join-Path $repo "dist\RootScope") `
        -DestinationPath $archive -CompressionLevel Optimal -Force
    $hash = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLowerInvariant()
    "$hash  RootScope-Windows-x64.zip" |
        Set-Content -Path (Join-Path $release "SHA256SUMS.txt") -Encoding ascii
    Write-Output "Windows release: $archive"
    Write-Output "SHA-256: $hash"
}
finally {
    Pop-Location
}
