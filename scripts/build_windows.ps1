param(
    [string]$PythonExe = ".\\.venv\\Scripts\\python.exe"
)

$ErrorActionPreference = "Continue"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$srcDir = Join-Path $repoRoot "src"
$mainPath = Join-Path $srcDir "main.py"
$modelPath = Join-Path $repoRoot "models/hand_landmarker.task"

if (Test-Path "bin") {
    Remove-Item "bin" -Recurse -Force
}

if (Test-Path "build") {
    Remove-Item "build" -Recurse -Force
}

if (Test-Path "release") {
    Remove-Item "release" -Recurse -Force
}

& $PythonExe -m PyInstaller --noconfirm --clean --onedir --name "MusicWithFingers" --paths $srcDir --distpath "bin" --workpath "build/work" --specpath "build/spec" --exclude-module "matplotlib" --exclude-module "tkinter" --add-data "$modelPath;models" $mainPath
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

$runtimeDir = Join-Path $repoRoot "bin/MusicWithFingers"
$exePath = Join-Path $runtimeDir "MusicWithFingers.exe"

if (-not (Test-Path $exePath)) {
    throw "Expected executable not found at $exePath"
}

New-Item -ItemType Directory -Path "release" -Force | Out-Null
$zipPath = Join-Path $repoRoot "release/MusicWithFingers-windows.zip"

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

& $PythonExe -c "from pathlib import Path; import shutil; root = Path(r'$repoRoot'); runtime = root / 'bin' / 'MusicWithFingers'; release = root / 'release'; release.mkdir(parents=True, exist_ok=True); archive_base = release / 'MusicWithFingers-windows'; zip_path = archive_base.with_suffix('.zip'); zip_path.unlink(missing_ok=True); shutil.make_archive(str(archive_base), 'zip', root_dir=str(runtime))"
if ($LASTEXITCODE -ne 0) {
    throw "Release zip creation failed with exit code $LASTEXITCODE"
}

Write-Host "Executable created: $exePath"
Write-Host "Distributable package created: $zipPath"
