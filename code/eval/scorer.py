"""Camera/class-balanced scoring and paired camera bootstrap, frozen rules."""
from collections import defaultdict
from fractions import Fraction
import argparse
import json
from pathlib import Path

def camera_scores(rows):
 classes=defaultdict(lambda:defaultdict(lambda:[0,0]))
 for row in rows:
  if row['truth'] is None:raise ValueError('fixed labelled score required')
  cell=classes[(row['dataset'],str(row['camera']))][row['truth']];cell[0]+=int(row['answer']==row['truth']);cell[1]+=1
 return {key:sum((Fraction(correct,total) for correct,total in values.values()),Fraction())/len(values) for key,values in classes.items()}

def summarize(edited,baseline):
 import numpy as np
 def index(rows):
  result={row['id']:row for row in rows}
  if len(result)!=len(rows):raise ValueError('duplicate paired crop ID')
  return result
 new,old=index(edited),index(baseline)
 if not old or set(new)!=set(old):raise ValueError('exact paired score IDs required')
 for key,row in old.items():
  if any(str(new[key][name])!=str(row[name]) for name in ('dataset','camera','truth')):raise ValueError('paired truth/camera identity differs')
 a,b=camera_scores(baseline),camera_scores(edited)
 if set(a)!=set(b):raise ValueError('paired camera coverage differs')
 keys=sorted(a);deltas=np.asarray([float(b[key]-a[key]) for key in keys],dtype=np.float64)
 draws=np.random.default_rng(20261112).integers(len(keys),size=(2000,len(keys)))
 interval=np.quantile(deltas[draws].mean(axis=1)*100,[.025,.975]).tolist()
 old_ba=float(sum(a.values(),Fraction())/len(a));new_ba=float(sum(b.values(),Fraction())/len(b))
 return dict(crops=len(old),cameras=len(keys),baseline_balanced_accuracy=old_ba,edited_balanced_accuracy=new_ba,
  paired_gain_pp=float(sum((b[k]-a[k] for k in keys),Fraction())/len(keys)*100),paired_camera_bootstrap_95ci_pp=interval,
  baseline_plain_accuracy=sum(r['answer']==r['truth'] for r in old.values())/len(old),
  edited_plain_accuracy=sum(r['answer']==r['truth'] for r in new.values())/len(new),
  fixes=sum(old[k]['answer']!=old[k]['truth'] and new[k]['answer']==new[k]['truth'] for k in old),
  harms=sum(old[k]['answer']==old[k]['truth'] and new[k]['answer']!=new[k]['truth'] for k in old),
  edited_nonanswers=sum(r['answer'] is None for r in new.values()),bootstrap_replicates=2000,bootstrap_seed=20261112,
  interval_scope='paired camera uncertainty conditional on this delivered edit; exploratory, not session-to-session variance')


def single(rows):
    scores = camera_scores(rows)
    if not scores or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique crop IDs required')
    return dict(crops=len(rows), cameras=len(scores),
                balanced_accuracy=float(sum(scores.values(), Fraction())/len(scores)),
                plain_accuracy=sum(r['answer']==r['truth'] for r in rows)/len(rows),
                nonanswers=sum(r['answer'] is None for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--edited',required=True);p.add_argument('--baseline');p.add_argument('--output',required=True)
    args=p.parse_args()
    def read(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows=read(args.edited)
    result=summarize(rows,read(args.baseline)) if args.baseline else single(rows)
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()
