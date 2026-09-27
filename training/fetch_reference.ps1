$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$revision = '506322f56b2ae1975e0c896fe7bccb731e698fac'
New-Item -ItemType Directory -Path '.reference' -Force | Out-Null
foreach ($sourceFile in @('src/Components/FishingGame.js', 'src/fishdata.js', 'README.md')) {
    $sourceName = Split-Path $sourceFile -Leaf
    Invoke-WebRequest -Uri ('https://raw.githubusercontent.com/abmasud1214/pufferdle/' + $revision + '/' + $sourceFile) -OutFile (Join-Path '.reference' $sourceName)
}
Write-Output 'Pinned numerical reference downloaded for local comparison.'
