"""Strict public manifest loader and frame-cache naming."""
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote

BASES={
 'ic':'https://lilawildlife.blob.core.windows.net/lila-wildlife/islandconservationcameratraps/public/',
 'caltech':'https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/'}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def cache_path(root,row):
    return Path(root)/row['dataset']/(row['image_sha256']+'.image')

def load(path):
    path=Path(path);manifest=json.loads(path.read_text())
    if manifest['format']!='camera-eval-v1' or manifest['crops']!=1611 or manifest['cameras']!=45:
        raise ValueError('expected fixed 1611-crop/45-camera manifest')
    relative=PurePosixPath(manifest['crops_file'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe crop manifest path')
    crops=path.parent/relative
    if sha(crops)!=manifest['crops_sha256']:raise ValueError('crop manifest hash differs')
    rows=[json.loads(line) for line in crops.read_text().splitlines() if line.strip()]
    if len(rows)!=1611 or len({r['id'] for r in rows})!=1611 or len({(r['dataset'],r['camera']) for r in rows})!=45:
        raise ValueError('crop/camera coverage differs')
    for row in rows:
        name=row['source_relative_path'];rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or any(v in name for v in ('\\','?','#')):raise ValueError('unsafe frame path')
        if row['url']!=BASES[row['dataset']]+quote(name,safe='/'):raise ValueError('unrecognized image URL')
        if len(row['image_sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['image_sha256']):raise ValueError('invalid frame SHA')
        if row['truth'] not in [o['label'] for o in row['options']]:raise ValueError('truth not in fixed options')
    if manifest.get('privacy',{}).get('ic_human_test_audit_passed') is not True:raise ValueError('IC privacy audit absent')
    return manifest,rows
