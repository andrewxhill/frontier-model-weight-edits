"""Download the source photos for an island crop manifest and cut the fixed crops.

Each row names a public photo URL (iNaturalist open data or LILA Island Conservation),
the SHA256 of the downloaded bytes and an integer box [x0, y0, x1, y1]. The crop is
taken from the raw decoded RGB pixels, with no EXIF rotation and no padding, and saved
as lossless PNG named by its pixel hash. Rows whose bytes or pixels differ are reported
and skipped; no image is fetched from any other source.

Example:
  python island/fetch_crops.py --manifest ../data/island/test.jsonl --output island-crops
"""
import argparse,hashlib,json,sys,time,urllib.request
from pathlib import Path

USER_AGENT='island-crop-fetch/1.0 (research reproduction)'

def pixel_hash(size,raw):
    header=json.dumps({'mode':'RGB','size':list(size)},sort_keys=True,allow_nan=False)+'\n'
    return hashlib.sha256(header.encode()+raw).hexdigest()

def crop_path(root,row):return Path(root)/(row['pixel_sha256']+'.png')

def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get('crop_status')=='complete']

def fetch(url,timeout=60):
    request=urllib.request.Request(url,headers={'User-Agent':USER_AGENT})
    with urllib.request.urlopen(request,timeout=timeout) as response:return response.read()

def cut(data,row):
    import io
    from PIL import Image
    with Image.open(io.BytesIO(data)) as image:rgb=image.convert('RGB')
    x0,y0,x1,y1=row['integer_box']
    if not(0<=x0<x1<=rgb.width and 0<=y0<y1<=rgb.height):raise ValueError('box outside image')
    crop=rgb.crop((x0,y0,x1,y1))
    if pixel_hash(crop.size,crop.tobytes())!=row['pixel_sha256']:raise ValueError('crop pixels differ')
    return crop

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest',required=True,help='island crop manifest (JSON lines)')
    p.add_argument('--output',required=True,help='directory for <pixel_sha256>.png crops')
    p.add_argument('--limit',type=int,default=None,help='fetch at most this many rows')
    p.add_argument('--delay',type=float,default=1.1,help='seconds between downloads')
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=read_rows(a.manifest)[:a.limit];failed=[]
    for i,row in enumerate(rows):
        target=crop_path(out,row)
        if target.exists():continue
        try:
            data=fetch(row['url'])
            if hashlib.sha256(data).hexdigest()!=row['source_image_sha256']:raise ValueError('downloaded bytes differ')
            cut(data,row).save(target,format='PNG')
        except Exception as error:
            failed.append(dict(id=row['id'],error=str(error)));print('skip',row['id'],error,file=sys.stderr)
        time.sleep(a.delay)
    print(json.dumps(dict(rows=len(rows),failed=len(failed)),indent=1))
    if failed:(out/'failures.json').write_text(json.dumps(failed,indent=1)+'\n')

if __name__=='__main__':main()
