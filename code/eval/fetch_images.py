"""Download the public LILA frames listed in the camera manifest and check each SHA256 hash."""
import argparse
from pathlib import Path
import urllib.request
from manifest import load,cache_path,sha


def fetch(manifest_path,root):
    _,rows=load(manifest_path);done=set()
    for row in rows:
        target=cache_path(root,row)
        if target in done:continue
        if target.exists():
            if target.is_symlink() or sha(target)!=row['image_sha256']:raise ValueError('cached frame SHA differs')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.partial')
            owned=False
            try:
                with urllib.request.urlopen(row['url'],timeout=60) as response,temporary.open('xb') as out:
                    owned=True
                    while block:=response.read(1024*1024):out.write(block)
                if sha(temporary)!=row['image_sha256']:raise ValueError('downloaded frame SHA differs')
                temporary.replace(target)
            finally:
                if owned and temporary.exists():temporary.unlink()
        done.add(target)
    return len(done)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,help='camera manifest, ../data/camera/manifest.json');p.add_argument('--images',required=True,help='directory to store the frames')
    a=p.parse_args();print('Verified frames:',fetch(a.manifest,a.images))
