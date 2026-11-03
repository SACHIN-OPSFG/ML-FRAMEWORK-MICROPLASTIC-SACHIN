from pathlib import Path
import argparse
from ultralytics import YOLO
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('image'); parser.add_argument('--confidence',type=float,default=.25)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    YOLO(str(root/'ml/models/best.pt')).predict(args.image,conf=args.confidence,save=True,project=str(root/'ml/predictions'),name='sample',exist_ok=True)
