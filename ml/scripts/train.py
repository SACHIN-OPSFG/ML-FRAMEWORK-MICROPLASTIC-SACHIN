from pathlib import Path
import argparse, json, shutil, math
import torch, yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

def train(epochs=40, imgsz=640, batch=12):
    data = ROOT/'ml/data/data.yaml'
    config = yaml.safe_load(data.read_text()); config['path'] = str(data.parent.resolve())
    data.write_text(yaml.safe_dump(config))
    model = YOLO('yolov8n.pt')
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                device=0 if torch.cuda.is_available() else 'cpu', workers=4,
                project=str(ROOT/'ml/runs'), name='fragment_detector', exist_ok=True,
                seed=42, deterministic=True, patience=12, plots=True, cache=False,
                close_mosaic=8, amp=False)
    target = ROOT/'ml/models'; target.mkdir(parents=True, exist_ok=True)
    for name in ['best.pt','last.pt']:
        source=ROOT/'ml/runs/fragment_detector/weights'/name
        checkpoint = torch.load(source,weights_only=False,map_location='cpu')
        fitted = checkpoint.get('ema') or checkpoint.get('model')
        if not all(torch.isfinite(v).all() for v in fitted.state_dict().values()):
            raise RuntimeError('Non-finite model weights; do not use this model.')
        shutil.copy2(source, target/name)
    evaluate()

def evaluate():
    data = ROOT/'ml/data/data.yaml'
    config = yaml.safe_load(data.read_text()); config['path'] = str(data.parent.resolve())
    data.write_text(yaml.safe_dump(config))
    model = YOLO(str(ROOT/'ml/models/best.pt'))
    output = {}
    for split in ['val','test']:
        metric = model.val(data=str(ROOT/'ml/data/data.yaml'),split=split,imgsz=640,batch=12,
                           device=0 if torch.cuda.is_available() else 'cpu',workers=0,
                           project=str(ROOT/'ml/evaluation'),name=split,exist_ok=True,plots=True)
        output[split] = {k:float(v) for k,v in metric.results_dict.items()}
        if not all(math.isfinite(v) for v in output[split].values()):
            raise RuntimeError('Evaluation produced non-finite metrics.')
        output[split]['speed_ms_per_image'] = metric.speed
    (ROOT/'ml/evaluation/metrics.json').write_text(json.dumps(output,indent=2))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--imgsz',type=int,default=640); parser.add_argument('--batch',type=int,default=12)
    args=parser.parse_args(); train(args.epochs,args.imgsz,args.batch)
