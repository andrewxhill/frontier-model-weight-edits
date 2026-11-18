"""Download the practice frames the recipe selects and check each SHA256 hash.

Downloads one frame at a time from the public LILA URLs. A frame already in the
cache with the right hash is reused.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tempfile
import urllib.request
from urllib.parse import quote

BASES={'ic':'https://lilawildlife.blob.core.windows.net/lila-wildlife/islandconservationcameratraps/public/',
       'caltech':'https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/'}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fetch_rows(rows,selected,images,*,opener=urllib.request.urlopen):
    mapping={r['key']:r for r in rows}
    if len(mapping)!=len(rows) or not selected<=set(mapping):raise ValueError('selected public practice IDs missing/duplicated')
    images=Path(images)
    if images.is_symlink():raise ValueError('image cache root may not be a symlink')
    done=set();downloaded=0
    for key in sorted(selected):
        row=mapping[key];name=row['source_relative_path'];relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or any(c in name for c in ('\\','?','#')) or relative.suffix.lower() not in ('.jpg','.jpeg','.png'):raise ValueError('unsafe public image path')
        if row['url']!=BASES[row['dataset']]+quote(name,safe='/'):raise ValueError('unrecognized public source URL')
        pin=row['image_sha256']
        if len(pin)!=64 or any(c not in '0123456789abcdef' for c in pin):raise ValueError('invalid source-frame SHA')
        target=images/row['dataset']/(pin+'.image')
        if target in done:continue
        if target.exists():
            if target.is_symlink() or sha(target)!=pin:raise ValueError('existing public frame cache differs')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.parent.is_symlink():raise ValueError('dataset cache directory may not be a symlink')
            temporary=None
            try:
                # This invocation owns only the uniquely created file below.
                with tempfile.NamedTemporaryFile(prefix=pin+'.',suffix='.partial',dir=target.parent,delete=False) as stream:
                    temporary=Path(stream.name)
                    with opener(row['url'],timeout=60) as response:
                        while block:=response.read(1024*1024):stream.write(block)
                if sha(temporary)!=pin:raise ValueError('downloaded frame SHA differs')
                if target.exists():
                    if target.is_symlink() or sha(target)!=pin:raise ValueError('concurrent cache target differs')
                else:temporary.replace(target);downloaded+=1
            finally:
                if temporary is not None and temporary.exists():temporary.unlink()
        done.add(target)
    return dict(selected_crops=len(selected),verified_frames=len(done),downloaded_frames=downloaded)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--practice',required=True,help='practice crop list, ../data/camera/practice/crops.json');p.add_argument('--selections',required=True,help='recipe/selections047.json');p.add_argument('--images',required=True,help='directory to store the frames');a=p.parse_args()
    manifest=json.loads(Path(a.practice).read_text());selection=json.loads(Path(a.selections).read_text())
    if manifest.get('format')!='public-practice-crops-v1' or manifest.get('privacy',{}).get('passed') is not True:raise ValueError('privacy-audited public practice manifest required')
    if sha(a.selections)!=manifest['sources']['selections_sha256']:raise ValueError('fixed recorded selection SHA differs')
    wanted=set(k for group in selection['groups'].values() for k in group['ids'])
    if len(wanted)!=2449:raise ValueError('fixed construction selection coverage differs')
    print(json.dumps(fetch_rows(manifest['rows'],wanted,a.images),indent=2))
