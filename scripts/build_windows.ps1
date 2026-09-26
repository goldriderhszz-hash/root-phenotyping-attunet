param(
    [string]$Python = "python",
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $repo
try {
    if (-not $SkipBuild) {
        & $Python -m PyInstaller --noconfirm --clean --windowed --onedir `
            --name RootScope --collect-all tkinterdnd2 `
            --add-data "model;model" desktop.py
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
    }
    if (-not (Test-Path (Join-Path $repo "dist\RootScope\RootScope.exe"))) {
        throw "RootScope.exe not found; build before packaging"
    }

    $release = Join-Path $repo "release"
    New-Item -ItemType Directory -Path $release -Force | Out-Null
    $archive = Join-Path $release "RootScope-Desktop-Windows-x64-v1.1.0.zip"
    Compress-Archive -Path (Join-Path $repo "dist\RootScope") `
        -DestinationPath $archive -CompressionLevel Optimal -Force
    $hash = (Get-FileHash -Algorithm SHA256 -Path $archive).Hash.ToLowerInvariant()
    "$hash  RootScope-Desktop-Windows-x64-v1.1.0.zip" |
        Set-Content -Path (Join-Path $release "SHA256SUMS.txt") -Encoding ascii
    Write-Output "Windows release: $archive"
    Write-Output "SHA-256: $hash"
}
finally {
    Pop-Location
}
