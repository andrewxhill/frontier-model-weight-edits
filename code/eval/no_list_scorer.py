"""Score open-naming predictions with fixed rules and no model judge.

Strict scoring requires the same category as the reference label; lenient scoring
also accepts a declared ancestor or descendant. Both are reported.
"""
import json
import re
import unicodedata
from pathlib import Path

NONANSWERS=frozenset(('', 'unknown', 'uncertain', 'unsure', 'i cannot tell',
 'i cannot identify the animal', 'cannot tell', 'cannot determine', 'not sure',
 'no animal', 'no animals', 'empty', 'nothing', 'none', 'n/a'))

def normalize(value):
    if not isinstance(value,str):raise ValueError('answer must be text or null')
    value=unicodedata.normalize('NFKC',value).lower().strip()
    value=re.sub(r'[-\u2010-\u2015]', ' ',value)
    # Preserve separators, negation, and prose: whole-phrase equality is needed.
    value=value[:-1].rstrip()if value.endswith('.')else value
    value=re.sub(r'^(?:a|an|the)\s+','',value)
    return re.sub(r'\s+',' ',value)

def validate_taxonomy(taxonomy):
    datasets=taxonomy['classes_by_dataset'];aliases=taxonomy['aliases'];parents=taxonomy['parents']
    canonical={name for labels in datasets.values()for name in labels}
    if any(not isinstance(name,str)or not normalize(name) for name in canonical):raise ValueError('canonical names must be nonempty text')
    if any(len(labels)!=len(set(labels))for labels in datasets.values()):raise ValueError('duplicate dataset categories')
    if set(aliases)!=canonical:raise ValueError('alias table must cover exactly every category')
    lookup={normalize(name):name for name in canonical}
    if len(lookup)!=len(canonical):raise ValueError('normalized category collision')
    for name,variants in aliases.items():
        for variant in variants:
            key=normalize(variant)
            if not key or(key in lookup and lookup[key]!=name):raise ValueError('conflicting alias')
            lookup[key]=name
    for child,parent in parents.items():
        if child not in canonical or parent not in canonical or child==parent:raise ValueError('unknown or reflexive parent')
        seen={child};node=parent
        while node in parents:
            if node in seen:raise ValueError('cyclic taxonomy')
            seen.add(node);node=parents[node]
        if node in seen:raise ValueError('cyclic taxonomy')
    return lookup

def load_taxonomy(path):
    value=json.loads(Path(path).read_text());validate_taxonomy(value);return value

def _ancestor(ancestor,descendant,parents):
    while descendant in parents:
        descendant=parents[descendant]
        if descendant==ancestor:return True
    return False

def score_answer(answer,truth,dataset,taxonomy,*,fault=None):
    lookup=validate_taxonomy(taxonomy)
    if dataset not in taxonomy['classes_by_dataset']or truth not in taxonomy['classes_by_dataset'][dataset]:raise ValueError('truth outside frozen original dataset categories')
    if answer is not None and not isinstance(answer,str):raise ValueError('answer must be text or null')
    normalized=None if answer is None else normalize(answer)
    prediction=lookup.get(normalized)
    nonanswer=bool(fault)or answer is None or normalized in NONANSWERS
    strict=not nonanswer and prediction==truth
    lenient=not nonanswer and prediction is not None and(strict or _ancestor(prediction,truth,taxonomy['parents'])or _ancestor(truth,prediction,taxonomy['parents']))
    status='fault'if fault else'nonanswer'if nonanswer else'mapped_answer'if prediction is not None else'unmapped_answer'
    return dict(normalized=normalized,prediction=prediction,strict=bool(strict),lenient=bool(lenient),
        nonanswer=bool(nonanswer),status=status,fault=fault)


def open_fault(raw,tokens,max_tokens):
    from parser import classify_reply
    fault=classify_reply(raw,tokens,[],max_tokens=max_tokens)
    return None if fault=='malformed' else fault


def summarize_rows(rows,taxonomy):
    from scorer import single
    result={}
    for mode in ('strict','lenient'):
        scored=[]
        for row in rows:
            outcome=score_answer(row['raw'],row['truth'],row['dataset'],taxonomy,fault=row.get('fault'))
            scored.append(dict(row,answer=row['truth'] if outcome[mode] else None))
        result[mode]=single(scored)
        result[mode]['nonanswers']=sum(score_answer(row['raw'],row['truth'],row['dataset'],taxonomy,fault=row.get('fault'))['nonanswer'] for row in rows)
    return result


def paired_rows(edited,baseline,taxonomy):
    from scorer import summarize
    result={}
    for mode in ('strict','lenient'):
        def project(rows):
            return [dict(r,answer=r['truth'] if score_answer(r['raw'],r['truth'],r['dataset'],taxonomy,fault=r.get('fault'))[mode] else None) for r in rows]
        result[mode]=summarize(project(edited),project(baseline))
        result[mode]['edited_nonanswers']=sum(score_answer(r['raw'],r['truth'],r['dataset'],taxonomy,fault=r.get('fault'))['nonanswer'] for r in edited)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--edited',required=True,help='predictions.jsonl for the model to score');parser.add_argument('--baseline',help='predictions.jsonl for a model to compare against');parser.add_argument('--taxonomy',default=str(Path(__file__).with_name('no-list-taxonomy.json')),help='aliases and parent-child rules');parser.add_argument('--output',required=True,help='JSON summary to write')
    a=parser.parse_args()
    def read(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    taxonomy=load_taxonomy(a.taxonomy)
    rows=read(a.edited)
    result=paired_rows(rows,read(a.baseline),taxonomy) if a.baseline else summarize_rows(rows,taxonomy)
    Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
