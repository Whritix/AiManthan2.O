[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, Position = 0)]
    [Alias("VideoPath")]
    [string]$InputPath,
    [string]$Profile = "steel",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$resolvedInput = Resolve-Path -LiteralPath $InputPath -ErrorAction Stop
$inputFile = Get-Item -LiteralPath $resolvedInput
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$videoExtensions = @(".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
$imageExtensions = @(".jpg", ".jpeg", ".png", ".bmp", ".webp")
$extension = $inputFile.Extension.ToLowerInvariant()
if ($extension -notin ($videoExtensions + $imageExtensions)) {
    throw "Unsupported input format '$extension'. Use JPG, PNG, BMP, WEBP, MP4, AVI, MOV, MKV, WEBM, or M4V."
}
$source = if ($extension -in $imageExtensions) { "image" } else { "video" }

$weights = Join-Path $ProjectRoot "artifacts\models\detector_best.pt"
$onnxWeights = Join-Path $ProjectRoot "artifacts\models\detector_best.onnx"
if (-not (Test-Path -LiteralPath $weights) -and -not (Test-Path -LiteralPath $onnxWeights)) {
    throw "Promoted YOLO weights are missing. Expected detector_best.pt or detector_best.onnx under artifacts\models."
}

$portProbe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
try {
    $portProbe.Start()
} catch {
    throw "Port $Port is already in use. Stop the existing server, then run this launcher again."
} finally {
    $portProbe.Stop()
}

$env:INDORE_PROFILE = $Profile
$env:INDORE_SOURCE = $source
if ($source -eq "image") { $env:INDORE_IMAGE_PATH = $inputFile.FullName }
else { $env:INDORE_VIDEO_PATH = $inputFile.FullName }
$env:INDORE_DETECTOR = "real"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot "Ultralytics"

Write-Host "Checking real YOLO weights..." -ForegroundColor Cyan
python (Join-Path $ProjectRoot "scripts\check_real_detector.py") --input $inputFile.FullName
if ($LASTEXITCODE -ne 0) {
    throw "YOLO preflight failed; the demo server was not started."
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "web\dist\index.html"))) {
    throw "Built frontend is missing. From web\ run: npm install; npm run build"
}

Write-Host "Starting real YOLO $source demo" -ForegroundColor Green
Write-Host "Input: $($inputFile.FullName)"
Write-Host "Profile: $Profile"
Write-Host "Dashboard: http://127.0.0.1:$Port"
Write-Host "Health: http://127.0.0.1:$Port/api/health"
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
python -m uvicorn server.app.main:app --host 127.0.0.1 --port $Port --no-access-log
