# Deterministic reproduction

Extract VER3_CACHE_EXACT_COMPLETE_SOLUTION_001.zip. It requires Python 3.12 and PowerShell; the implementation uses the Python standard library only.

From the extracted solution directory, run:

~~~powershell
./RUN_REPRODUCE.ps1 -OutputRoot ../ver3-output
~~~

The output directory must be absent or empty. A successful run produces 55 predictions and verifies SHA-256 2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364.

The replay makes zero provider/API and evaluator calls and does not need credentials. The package includes the authoritative aggregate evaluation record. Official evaluator and gold contents are excluded; scores can be recomputed only in a separately provisioned environment whose evaluator and gold hashes match the recorded metadata.

