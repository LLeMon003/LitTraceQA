"""One-record local-model smoke with parent-paper source grounding."""
import argparse, hashlib, json, re, subprocess
from pathlib import Path

def norm(s): return ' '.join(str(s).split())
def terms(s): return {x for x in re.findall(r'[a-z0-9]+',s.lower()) if len(x)>1}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def rows(p):
    with p.open(encoding='utf8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--parent',type=Path,required=True); p.add_argument('--index',type=Path,required=True); p.add_argument('--query-id',required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()): raise FileExistsError(a.output_dir)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 q=next(x for x in rows(a.inputs) if x['query_id']==a.query_id); parent=next(x for x in rows(a.parent) if x['query_id']==a.query_id)
 papers={x.get('paper_id') for x in parent.get('gold_papers',[]) if isinstance(x,dict)}; qt=terms(q['question']); ranked=[]
 for x in rows(a.index):
  if x.get('paper_id') in papers:
   text=norm(x.get('text') or x.get('normalized_text') or ''); score=len(qt & terms(text))
   if text and score: ranked.append((score,x.get('object_uid'),x))
 ranked.sort(key=lambda x:(-x[0],str(x[1]))); evidence=[{'object_uid':x[1],'page':x[2].get('page'),'object_type':x[2].get('object_type'),'text':norm(x[2].get('text') or x[2].get('normalized_text'))[:500]} for x in ranked[:3]]
 prompt=json.dumps({'instruction':'Return JSON only: {"answer":"...","evidence_object_id":"...","quote":"...","confidence":0..1}. Use only an exact quote from one object; otherwise answer empty.','question':q['question'],'evidence':evidence},ensure_ascii=False)
 r=subprocess.run(['ollama','run','qwen3-vl:4b',prompt],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90); raw=r.stdout
 try: out=json.loads(raw); chosen=next((x for x in evidence if x['object_uid']==out.get('evidence_object_id')),None); grounded=bool(chosen and norm(out.get('quote','')) and norm(out['quote']) in chosen['text'] and norm(out.get('answer','')) in chosen['text'])
 except Exception: out={}; grounded=False
 (a.output_dir/'raw_response.txt').write_text(raw,encoding='utf8'); summary={'query_id':a.query_id,'parent_papers':sorted(papers),'evidence_count':len(evidence),'grounded':grounded,'response':out if grounded else {'status':'rejected'},'inputs_sha256':sha(a.inputs),'parent_sha256':sha(a.parent),'index_sha256':sha(a.index),'raw_sha256':sha(a.output_dir/'raw_response.txt'),'official_gold_used':False,'candidate_written':False}
 for name in ('status.json','manifest.json'):
  t=a.output_dir/(name+'.tmp'); t.write_text(json.dumps({'status':'complete',**summary},sort_keys=True)+'\n',encoding='utf8'); t.replace(a.output_dir/name)
 print(json.dumps({'query_id':a.query_id,'evidence_count':len(evidence),'grounded':grounded,'raw_sha256':summary['raw_sha256']}))
if __name__=='__main__': main()
