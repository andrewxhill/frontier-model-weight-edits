"""Public-practice metadata adapter. No model or image imports at module import."""
import hashlib
import json
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def select_rows(rows,ids,require_labels=False):
    mapping={row['key']:row for row in rows}
    if len(mapping)!=len(rows) or len(ids)!=len(set(ids)):raise ValueError('duplicate practice/source selection identity')
    selected=[]
    for key in ids:
        if key not in mapping:raise ValueError('selected ID outside public practice')
        row=mapping[key]
        if key!=row['dataset']+'/'+str(row['id'])+':'+str(row['box_index']):raise ValueError('practice crop identity differs')
        if require_labels and row['label'] is None:raise ValueError('unknown public label cannot enter a supervised fit')
        selected.append(row)
    return selected


def read_practice(path,cameras):
    value=json.loads(Path(path).read_text())
    rows=value['rows'] if isinstance(value,dict) else value
    if isinstance(value,dict) and 'keys' in value:
        if len(value['keys'])!=len(rows):raise ValueError('public row/key lengths differ')
        rows=[dict(row,key=key) for row,key in zip(rows,value['keys'])]
    camera_table=json.loads(Path(cameras).read_text())
    if len(rows)!=23447 or len(camera_table)!=57:raise ValueError('original public practice coverage required')
    if sum(row['label'] is None for row in rows)!=308:raise ValueError('original unknown-label roster required')
    for row in rows:
        names=camera_table[row['dataset']+'/'+str(row['location'])]
        if row['label'] is not None and row['label'] not in names:raise ValueError('public practice label absent from own-camera options')
        if row['label'] is not None and row['label'].casefold() in ('human','person','people'):raise ValueError('human label is not an admitted practice animal')
    select_rows(rows,[r['key'] for r in rows])
    return rows,camera_table


def selections(path,rows):
    value=json.loads(Path(path).read_text())
    if value['format']!='recorded-practice-selections-v1':raise ValueError('recorded selection format required')
    return {name:select_rows(rows,entry['ids']) for name,entry in value['groups'].items()},value['dev_cameras']


def frame_path(images,row):
    return Path(images)/row['dataset']/(row['image_sha256']+'.image')


def options(names):
    result=[]
    for index,name in enumerate(sorted(names)):
        n=index+1;letter=''
        while n:n,k=divmod(n-1,26);letter=chr(65+k)+letter
        result.append(dict(label=name,name=name,letter=letter,index=index+1))
    return result
