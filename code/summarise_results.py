"""Recompute every headline score from the saved per-crop predictions in results/.

Camera scores give each of the 45 cameras equal weight, and each animal class within a
camera equal weight. Paired intervals resample whole cameras (2,000 draws, seed 20261112).
Island scores use island/score.py on the saved answers.

Example:
  python summarise_results.py --results ../results --data ../data --output ../results/summary.json
"""
import argparse,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'eval'));sys.path.insert(0,str(HERE/'island'))
MODELS=['unedited','camera-trap-edit','galapagos-island-edit','camera-island-direct-edit','camera-island-lora']
CAMERA_PAIRS=[('camera-trap-edit','unedited'),('galapagos-island-edit','unedited'),('camera-island-direct-edit','unedited'),('camera-island-lora','unedited'),('camera-island-lora','camera-island-direct-edit'),('camera-island-lora','camera-trap-edit'),('camera-island-direct-edit','camera-trap-edit')]
ISLAND_READINGS=MODELS+['matched-siglip','speciesnet-stock','speciesnet-raw']

def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def build(results,data):
    import scorer,score
    results=Path(results);camera={};paired={}
    for name in MODELS+['matched-siglip']:
        path=results/'camera'/(name+'.jsonl')
        if path.exists():
            s=scorer.single(read(path));camera[name]=dict(balanced_accuracy=round(100*s['balanced_accuracy'],2),plain_accuracy=round(100*s['plain_accuracy'],2),crops=s['crops'],cameras=s['cameras'],nonanswers=s['nonanswers'])
    for new,old in CAMERA_PAIRS:
        s=scorer.summarize(read(results/'camera'/(new+'.jsonl')),read(results/'camera'/(old+'.jsonl')))
        paired[new+' minus '+old]=dict(change_pp=round(s['paired_gain_pp'],2),interval_95_pp=[round(v,2) for v in s['paired_camera_bootstrap_95ci_pp']],fixes=s['fixes'],harms=s['harms'])
    rows={r['id']:r for r in read(Path(data)/'island'/'test.jsonl') if r.get('crop_status')=='complete'}
    lists=json.loads((HERE/'island'/'menus.json').read_text())['lists'];resolve=score.build_lookup(lists);island={}
    for name in ISLAND_READINGS:
        saved=read(results/'island'/(name+'.jsonl'))
        if name.startswith('speciesnet') or name=='matched-siglip':
            evaluated=[dict(id=p['id'],group=p['group'],domain=p['domain'],location=p['location'],native_status_correct=p['native_status_correct'],genus_correct=p['genus_correct']) for p in saved]
        else:
            evaluated=[score.evaluate(p,rows[p['id']],resolve) for p in saved]
        island[name]={k:{f:(round(v,2) if isinstance(v,float) else v) for f,v in d.items()} for k,d in score.summarise(evaluated).items()}
    return dict(camera_benchmark=camera,camera_paired_changes=paired,island_test=island,
        notes=dict(camera='1,611 crops from 45 cameras, camera-specific option lists',
            island='697 crops; native contrasts use public photographs; Galapagos natives come from one public county per group',
            readers='SpeciesNet and SigLIP island rows are re-aggregated from their saved per-crop decisions, which used the SpeciesNet taxonomy release'))

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results',default=str(HERE.parent/'results'),help='folder of saved predictions')
    p.add_argument('--data',default=str(HERE.parent/'data'),help='folder of data manifests')
    p.add_argument('--output',required=True,help='JSON summary to write')
    a=p.parse_args();summary=build(a.results,a.data)
    Path(a.output).write_text(json.dumps(summary,indent=1)+'\n');print(json.dumps(summary['island_test']['camera-island-lora'],indent=1))

if __name__=='__main__':main()
