"""Score island predictions: native-versus-look-alike balanced accuracy and rat recall.

Each answer is resolved to a taxon through the list names and scientific names, and,
when supplied, the SpeciesNet taxonomy release used in the study. Scores give each
public location equal weight within a direction, then average the two directions:
  land iguana vs green iguana, marine iguana vs green iguana,
  Galapagos mockingbird vs brown noddy (photographs only),
  plus equal-location exact-genus recall for invasive Rattus.

Example:
  python island/score.py --manifest ../data/island/test.jsonl \
    --predictions island-edit-predictions.jsonl --output island-scores.json
"""
import argparse,collections,json,re,unicodedata
from fractions import Fraction
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPECIES_SYNONYMS={'capra aegagrus hircus':'capra hircus','sus scrofa domesticus':'sus scrofa'}
NONANSWER={'','unknown','unsure','cannot tell','cannot determine','no animal','no animals','none','empty','blank'}
GENUS_FAMILIES={'nesoryzomys':('mammalia','cricetidae'),'aegialomys':('mammalia','cricetidae'),'rattus':('mammalia','muridae'),'mus':('mammalia','muridae'),'conolophus':('reptilia','iguanidae'),'amblyrhynchus':('reptilia','iguanidae'),'iguana':('reptilia','iguanidae'),'mimus':('aves','mimidae'),'zenaida':('aves','columbidae'),'setophaga':('aves','parulidae'),'sula':('aves','sulidae'),'anous':('aves','laridae'),'pterodroma':('aves','procellariidae'),'felis':('mammalia','felidae'),'nasua':('mammalia','procyonidae'),'capra':('mammalia','bovidae'),'sus':('mammalia','suidae'),'oryctolagus':('mammalia','leporidae')}
CONTRASTS={'land_iguana_vs_green_iguana':('native_land_iguana','green_iguana'),'marine_iguana_vs_green_iguana':('native_marine_iguana','green_iguana'),'mockingbird_vs_brown_noddy':('native_mockingbird','other_birds')}

def norm(x):
    x=''.join(c for c in unicodedata.normalize('NFKD',x).casefold() if not unicodedata.combining(c))
    x=re.sub(r'[-‐-―]',' ',x);return re.sub(r'\s+',' ',x).strip(' .\t\r\n')
def canonical_species(x):return SPECIES_SYNONYMS.get(norm(x),norm(x))

def label_node(label):
    fields=label.strip().split(';')
    if len(fields)!=7:raise ValueError('taxonomy label must have seven fields')
    uuid,cl,order,fam,genus,sp,common=fields
    return dict(uuid=uuid,label=label,class_name=cl,family=fam,order=order,genus=genus,species=(genus+' '+sp).strip() if genus and sp else '',common=common)

def build_lookup(lists,taxonomy_lines=()):
    aliases={}
    def add(key,node):
        if not key:return
        signature=(node.get('class_name',''),node.get('family',''),node.get('genus',''),canonical_species(node.get('species','')))
        aliases.setdefault(norm(key),{})[signature]=node
    for line in taxonomy_lines:
        if not line.strip():continue
        node=label_node(line);add(node['common'],node)
        if node['species']:add(node['species'],node)
        elif node['genus']:add(node['genus'],node)
    for menu in lists.values():
        for o in menu:
            genus=o['genus'];cl,fam=GENUS_FAMILIES[genus]
            node=dict(class_name=cl,family=fam,genus=genus,species=norm(o['scientific']),common=o['name'])
            add(o['name'],node);add(o['scientific'],node);add(genus,dict(class_name=cl,family=fam,genus=genus,species='',common=''))
    for word,node in {'rat':dict(class_name='mammalia',family='',genus='',species=''),'rice rat':dict(class_name='mammalia',family='cricetidae',genus='',species=''),'iguana':dict(class_name='reptilia',family='iguanidae',genus='',species=''),'mockingbird':dict(class_name='aves',family='mimidae',genus='',species=''),'bird':dict(class_name='aves',family='',genus='',species='')}.items():
        aliases[norm(word)]={('explicit_coarse',):node}
    def resolve(value):
        if not isinstance(value,str):return None
        choices=aliases.get(norm(value),{})
        return next(iter(choices.values())) if len(choices)==1 else None
    return resolve

def group_correct(node,row):
    if not node:return False
    group=row['group'];genus=node.get('genus','')
    rule={'native_rice_rat':lambda:genus in ('nesoryzomys','aegialomys'),'invasive_rattus':lambda:genus=='rattus','native_land_iguana':lambda:genus=='conolophus','native_marine_iguana':lambda:genus=='amblyrhynchus','green_iguana':lambda:genus=='iguana','native_mockingbird':lambda:genus=='mimus'}
    if group in rule:return rule[group]()
    if group=='other_birds':
        if node.get('class_name')!='aves':return False
        if genus:return genus!='mimus'
        if node.get('family'):return node['family']!='mimidae'
        return bool(node.get('order')) and node['order']!='passeriformes'
    return bool(node.get('species')) and canonical_species(node['species'])==canonical_species(row['truth_species'])

def evaluate(prediction,row,resolve):
    """Native status: either native iguana genus counts as native; other groups use the group rule."""
    node=None
    if prediction.get('fault') in (None,'malformed'):node=resolve(prediction.get('answer')) or resolve(prediction.get('raw',''))
    native=bool(node and node.get('genus') in ('conolophus','amblyrhynchus')) if row['group'] in ('native_land_iguana','native_marine_iguana') else group_correct(node,row)
    return dict(id=row['id'],group=row['group'],domain=row['domain'],location=row['location'],answer=prediction.get('answer'),
        genus_correct=bool(node and node.get('genus')==row['truth_genus']),native_status_correct=native,mapped_node=node)

def equal_location(rows,key):
    by=collections.defaultdict(list)
    for r in rows:by[r['location']].append(bool(r[key]))
    if not by:return None
    return float(sum((Fraction(sum(v),len(v)) for v in by.values()),Fraction())/len(by))

def summarise(evaluated):
    photos=[r for r in evaluated if r['domain']=='citizen_photo'];out={}
    for name,(native,other) in CONTRASTS.items():
        a=equal_location([r for r in photos if r['group']==native],'native_status_correct');b=equal_location([r for r in photos if r['group']==other],'native_status_correct')
        out[name]=dict(balanced_accuracy=None if a is None or b is None else 100*(a+b)/2,native_recall=None if a is None else 100*a,lookalike_recall=None if b is None else 100*b,
            crops=sum(r['group'] in (native,other) for r in photos))
    rats=[r for r in evaluated if r['group']=='invasive_rattus'];recall=equal_location(rats,'genus_correct')
    out['invasive_rat_recall']=dict(equal_location_recall=None if recall is None else 100*recall,crops=len(rats),correct=sum(r['genus_correct'] for r in rats))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest',required=True,help='island test manifest (JSON lines)')
    p.add_argument('--predictions',required=True,help='output of predict.py')
    p.add_argument('--output',required=True,help='JSON summary to write')
    p.add_argument('--menus',default=str(HERE/'menus.json'),help='the three 11-name lists')
    p.add_argument('--speciesnet-taxonomy',default=None,help='optional SpeciesNet taxonomy release file (one 7-field label per line), used in the study to resolve extra common names')
    a=p.parse_args()
    lists=json.loads(Path(a.menus).read_text())['lists']
    taxonomy=Path(a.speciesnet_taxonomy).read_text().splitlines() if a.speciesnet_taxonomy else ()
    resolve=build_lookup(lists,taxonomy)
    rows={r['id']:r for r in (json.loads(x) for x in Path(a.manifest).read_text().splitlines() if x.strip()) if r.get('crop_status')=='complete'}
    predictions=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]
    missing=set(rows)-{p['id'] for p in predictions}
    evaluated=[evaluate(p,rows[p['id']],resolve) for p in predictions if p['id'] in rows]
    summary=dict(crops=len(evaluated),missing_predictions=len(missing),scores=summarise(evaluated))
    Path(a.output).write_text(json.dumps(summary,indent=1)+'\n');print(json.dumps(summary,indent=1))

if __name__=='__main__':main()
