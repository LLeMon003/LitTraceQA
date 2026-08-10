param(
    [switch]$FetchOfficialSnapshot
)

$ErrorActionPreference = "Stop"
$releaseRoot = Split-Path -Parent $PSScriptRoot

Push-Location $releaseRoot
try {
    python scripts/build.py
    python scripts/verify_release.py --prediction build/test_predictions.jsonl
    if ($FetchOfficialSnapshot) {
        python scripts/fetch_official_snapshot.py
        python scripts/verify_release.py --prediction build/test_predictions.jsonl --official-snapshot .cache/official_snapshot
    }
}
finally {
    Pop-Location
}
