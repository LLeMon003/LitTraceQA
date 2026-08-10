"""Score answer-bearing internal retrieval on a frozen target-aligned benchmark."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.target_retrieval import score

def rows(paths):
    for path in paths:
        for line in Path(path).open(encoding='utf8'):
            if line.strip(): yield json.loads(line)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--benchmark',type=Path,required=True); p.add_argument('--facts',type=Path,required=True); p.add_argument('--table-ledger',type=Path,action='append',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True,exist_ok=True); benchmark=list(rows([a.benchmark])); result=score(benchmark, rows([a.facts,*a.table_ledger]))
    gate={'overall_at_least_0_85':result['answer_bearing_recall']>=.85,'family_minimum_0_70':all(v>=.70 for v in result['family_recall'].values())}
    report={'status':'complete','benchmark_path':str(a.benchmark),'result':result,'gate':gate,'classification':'PASS' if all(gate.values()) else 'FAILED_RETRIEVAL_GATE','official_gold_used':False,'candidate_written':False}
    for name in ('status.json','result.json'):
        tmp=a.output_dir/(name+'.tmp'); tmp.write_text(json.dumps(report,sort_keys=True)+'\n',encoding='utf8'); tmp.replace(a.output_dir/name)
    print(json.dumps({'classification':report['classification'],'recall':result['answer_bearing_recall'],'family_recall':result['family_recall']}))
if __name__=='__main__': main()
