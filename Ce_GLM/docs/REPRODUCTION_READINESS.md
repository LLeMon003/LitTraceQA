# Version 2.3 reproduction readiness

Primary classification: `FRESH_PIPELINE_COMPLETE_METRICS_DIVERGENT`.

## Separate reproducibility dimensions

| Dimension | Status | Evidence |
|---|---|---|
| Cached deterministic DEV | Exact | SHA-256 `2635F99AA8B1C22AD941AB2761C64622710528DA833A16014FF7DF021EF69364` |
| Cached deterministic production | Exact | SHA-256 `076C949F4B40FDBF8D963CA5A92C31838C9F24D6BE9D7ED97ABC69DCEC92AFF8` |
| Integrated code | Complete | Single entry point supports cached DEV, cached production, raw fresh, and resumable execution; accepted cache boundary is documented |
| Fresh generation | Complete | 55 records; 30 preserved and 25 newly generated; frozen reassembled output |
| Official evaluation | Complete | Verified gold/evaluator/prediction hashes; one authoritative invocation |
| Metric reproduction | Materially divergent | Paper exact; Evidence, MC, Freeform, and Table metrics below cached-exact |
| Artifact completeness | Complete with historical lineage limitation | Gold, raw logs, caches, and cache-boundary records exist; the upstream producer edge for the historical base cache remains unverified |

The fresh run is a complete raw API reproduction of the packaged Version 2 pipeline. That does not erase the documented historical cache-boundary limitation: exact deterministic replay below the accepted cache is proven, while the historical producer edge into that cache is not.

## Authoritative metric comparison

| Metric | Fresh | Cached-exact DEV | Delta |
|---|---:|---:|---:|
| Paper F1 | 1.000000 | 1.000000 | 0.000000 |
| Evidence precision | 0.618254 | 0.684343 | -0.066089 |
| Evidence recall | 0.568687 | 0.620707 | -0.052020 |
| Evidence F1 | 0.581560 | 0.634646 | -0.053087 |
| MC accuracy | 27/41 | 38/41 | -11 answers |
| Freeform exact match | 16/26 | 23/26 | -7 answers |
| Table row F1 | 0.875397 | 1.000000 | -0.124603 |
| Table-cell macro | 0.573232 | 0.954545 | -0.381313 |
| Table-cell micro | 14/27 | 25/27 | -11 cells |

These differences are genuine fresh-generation and downstream-stage reproducibility gaps. They are not deterministic replay failures and were not used to change the prediction.

## Evaluation status labels

- Raw question input used as gold: `INVALID`.
- Cached prediction used as gold: `DIAGNOSTIC_ONLY`.
- Hash-locked `LitTraceQA/data/validation.jsonl` evaluation: `AUTHORITATIVE`.

The package is ready for reproducibility review with the fresh metric divergence and historical cache boundary explicitly disclosed. It is not evidence that the historical best metrics can be regenerated exactly from a new third-party API sample.
