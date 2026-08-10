"""Audit the frozen Cycle 2 router using input-only held-out records."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hybrid_challenger import ProvenanceCoalescingHybridIndex
from src.locked_router import ROUTER_VERSION, route_input
from src.synthetic_benchmark import _canonical_json, _read_jsonl, sha256_file

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--facts',type=Path,required=True); p.add_argument('--table-ledger',type=Path,action='append',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    index=ProvenanceCoalescingHybridIndex(_read_jsonl([a.facts]),_read_jsonl(a.table_ledger)); rows=[route_input(r,index) for r in _read_jsonl([a.inputs])]
    out=a.output_dir/'routing_audit.jsonl'; tmp=out.with_suffix('.jsonl.tmp'); tmp.write_text(''.join(_canonical_json(r)+'\n' for r in rows),encoding='utf8'); tmp.replace(out)
    summary={'router_version':ROUTER_VERSION,'input_sha256':sha256_file(a.inputs),'record_count':len(rows),'selected_count':sum(r['selected'] for r in rows),'fallback_count':sum(not r['selected'] for r in rows),'audit_sha256':sha256_file(out),'official_gold_used':False,'parent_answers_read':False,'candidate_written':False}
    for name in ('status.json','manifest.json'):
      t=a.output_dir/(name+'.tmp'); t.write_text(_canonical_json({'status':'complete',**summary})+'\n',encoding='utf8'); t.replace(a.output_dir/name)
    print(_canonical_json(summary))
if __name__=='__main__': main()
