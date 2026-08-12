param(
    [switch]$ReuseSilent
)

$ErrorActionPreference = "Stop"

$demoRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $demoRoot
$tempRoot = Join-Path $demoRoot ".tmp"
$silentVideo = Join-Path $tempRoot "armbench-demo-silent.mp4"
$finalVideo = Join-Path $projectRoot "assets\armbench-demo.mp4"
$poster = Join-Path $projectRoot "assets\armbench-demo-poster.png"

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null

Push-Location $demoRoot
try {
    if (-not $ReuseSilent -or -not (Test-Path -LiteralPath $silentVideo)) {
        & pnpm exec remotion render src/index.ts ArmBenchDemo $silentVideo `
            --codec=h264 --crf=17 --pixel-format=yuv420p
        if ($LASTEXITCODE -ne 0) { throw "Remotion video render failed." }
    }
    else {
        Write-Host "Reusing silent render: $silentVideo"
    }

    & pnpm exec remotion still src/index.ts ArmBenchPoster $poster --frame=0
    if ($LASTEXITCODE -ne 0) { throw "Remotion poster render failed." }

    $tone = "aevalsrc=0.014*sin(2*PI*55*t)+0.006*sin(2*PI*82.41*t)+0.003*sin(2*PI*(110+0.7*sin(2*PI*0.04*t))*t):s=48000:d=74"
    $noise = "anoisesrc=color=pink:duration=74:amplitude=0.004:sample_rate=48000"
    & ffmpeg -y -hide_banner -loglevel warning `
        -f lavfi -i $tone -f lavfi -i $noise -i $silentVideo `
        -filter_complex "[0:a][1:a]amix=inputs=2:normalize=0,lowpass=f=1200,afade=t=in:st=0:d=2,afade=t=out:st=71:d=3,volume=0.55[a]" `
        -map 2:v:0 -map "[a]" `
        -vf "scale=in_range=pc:out_range=tv,format=yuv420p" `
        -c:v libx264 -preset medium -crf 17 `
        -c:a aac -b:a 160k -movflags +faststart -shortest $finalVideo
    if ($LASTEXITCODE -ne 0) { throw "FFmpeg audio mux failed." }

    & node scripts/verify-video.mjs
    if ($LASTEXITCODE -ne 0) { throw "Rendered artifact verification failed." }
}
finally {
    Pop-Location
}

Write-Host "Rendered: $finalVideo"
Write-Host "Poster:   $poster"
