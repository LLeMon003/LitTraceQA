param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$packageRoot = Split-Path -Parent $PSCommandPath
$artifactRoot = Join-Path $packageRoot 'sealed_bundle'
$runner = Join-Path $packageRoot 'release\scripts\portable_cache_exact_release.py'
$prediction = Join-Path $OutputRoot 'predictions.jsonl'
$expected = '2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364'

if (-not (Test-Path -LiteralPath $runner)) { throw "Missing packaged runner: $runner" }
if (-not (Test-Path -LiteralPath $artifactRoot)) { throw "Missing packaged sealed bundle: $artifactRoot" }
if ((Test-Path -LiteralPath $OutputRoot) -and (Get-ChildItem -LiteralPath $OutputRoot -Force | Select-Object -First 1)) {
    throw "OutputRoot must be absent or empty: $OutputRoot"
}

python $runner reproduce --artifact-root $artifactRoot --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { throw "Deterministic reproduction failed with exit code $LASTEXITCODE" }
if ((Get-FileHash -LiteralPath $prediction -Algorithm SHA256).Hash -ne $expected) {
    throw 'Prediction hash did not match the frozen Ver3 result.'
}

Write-Output "Reproduced prediction SHA-256: $expected"
