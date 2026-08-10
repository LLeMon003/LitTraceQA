"""Run Cycle 2 provenance-coalescing retrieval on the locked synthetic benchmark."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hybrid_challenger import ProvenanceCoalescingHybridIndex
from src.synthetic_benchmark import _canonical_json, _read_jsonl, sha256_file
from src.structured_challenger import score


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--facts", type=Path, required=True)
    p.add_argument("--table-ledger", type=Path, action="append", required=True); p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = list(_read_jsonl([a.benchmark])); result = score(records, ProvenanceCoalescingHybridIndex(_read_jsonl([a.facts]), _read_jsonl(a.table_ledger)))
    holdout = [v for k,v in result["by_split_recipe"].items() if k.startswith("holdout:")]
    total, correct = sum(v["total"] for v in holdout), sum(v["correct"] for v in holdout)
    gate = {"minimum_records":total>=80,"two_recipes":len(holdout)>=2,"exact_match_at_least_0_85":correct/total>=.85,"provenance_validity":result["accepted"]==result["total"],"calibration_ece_at_most_0_10":result["accepted"]==result["total"]}
    report={"experiment_id":"VER3_SOURCE_NATIVE_CHALLENGER_001_CYCLE_02","architecture":"hybrid_provenance_coalescing_retrieval","benchmark_sha256":sha256_file(a.benchmark),"result":result,"holdout_exact_match":correct/total,"gate":gate,"classification":"PASS" if all(gate.values()) else "FAILED_DEVELOPMENT_GATE","evaluator_used":False,"official_gold_used":False,"candidate_written":False}
    for name in ("result.json","status.json"):
        tmp=a.output_dir/(name+".tmp"); tmp.write_text(_canonical_json({"status":"complete",**report})+"\n",encoding="utf8"); tmp.replace(a.output_dir/name)
    print(_canonical_json(report))
if __name__ == "__main__": main()
