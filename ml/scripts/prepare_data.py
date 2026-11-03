"""Download the attributed CC BY 4.0 dataset and convert COCO boxes to YOLO."""
from pathlib import Path
import io, json, hashlib, time
import requests, yaml
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPO = 'andreusv/Microplastic-Fragments-WWTP-Dataset'
REVISION = 'bdbb9dd898b7d96187e1d7e4f1a49babc85f3160'
DATA = ROOT / 'ml/data'

def prepare():
    raw = DATA / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    card = requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md', timeout=60)
    card.raise_for_status(); (DATA / 'SOURCE_DATASET_CARD.md').write_text(card.text, encoding='utf-8')
    listing = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data', timeout=60)
    listing.raise_for_status(); meta = listing.json()
    stats = {}; hashes = {}; duplicates = []
    for item in meta:
        name = item['path']; target = raw / Path(name).name
        if not target.exists() or target.stat().st_size != item['size']:
            print('Downloading', name, flush=True)
            for attempt in range(4):
                try:
                    with requests.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}?download=true', stream=True, timeout=120) as response:
                        response.raise_for_status()
                        with target.with_suffix('.part').open('wb') as out:
                            for chunk in response.iter_content(1024*1024): out.write(chunk)
                    target.with_suffix('.part').replace(target); break
                except requests.RequestException:
                    if attempt == 3: raise
                    time.sleep(2)
        source_split = Path(name).name.split('-')[0]
        split = 'val' if source_split == 'validation' else source_split
        image_dir = DATA / split / 'images'; label_dir = DATA / split / 'labels'
        image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
        stat = stats.setdefault(split, {'images':0,'boxes':0,'background_images':0})
        for batch in pq.ParquetFile(target).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                blob = row['image']['bytes']; digest = hashlib.sha256(blob).hexdigest()
                if digest in hashes:
                    duplicates.append({'sha256':digest,'first':hashes[digest],'second':split})
                    continue
                hashes[digest] = split
                image = Image.open(io.BytesIO(blob)).convert('RGB'); w,h = image.size
                objects = row['objects']; labels = []
                for box, category in zip(objects['bbox'], objects['category_id']):
                    if category != 1: raise ValueError(f'Unexpected class {category}')
                    x,y,bw,bh = map(float,box)
                    x1,y1 = max(0,x),max(0,y); x2,y2 = min(w,x+bw),min(h,y+bh)
                    if x2<=x1 or y2<=y1: raise ValueError('Invalid box')
                    labels.append(f'0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
                stem = f'{split}_{stat["images"]:05d}'
                # Resize proportionally for portable distribution; normalized annotations remain valid.
                image.thumbnail((1280,1280)); image.save(image_dir / (stem+'.jpg'), quality=92)
                (label_dir / (stem+'.txt')).write_text('\n'.join(labels), encoding='utf-8')
                stat['images']+=1; stat['boxes']+=len(labels); stat['background_images']+=not labels
        print(split, stat, flush=True)
    config = {'path':str(DATA.resolve()),'train':'train/images','val':'val/images','test':'test/images','names':{0:'microplastic_fragment'},'nc':1}
    (DATA/'data.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    summary = {'source':f'https://huggingface.co/datasets/{REPO}','revision':REVISION,'license':'CC BY 4.0','stats':stats,'duplicates_removed':duplicates,'images_sha256':hashes,'conversion':'COCO xywh to normalized YOLO xywh; resized longest side <=1280'}
    (DATA/'dataset_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return stats

if __name__ == '__main__': prepare()
