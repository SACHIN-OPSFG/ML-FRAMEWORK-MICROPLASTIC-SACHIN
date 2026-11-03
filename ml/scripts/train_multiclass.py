"""
Multi-class microplastic detection model training script.
Trains a YOLOv8n model on the combined multi-class dataset.
Preserves the existing single-class baseline model.

Classes: fiber (0), fragment (1), film (2), pellet (3), foam (4), bead (5)
"""

from pathlib import Path
import argparse
import json
import hashlib
import time
import math
import shutil
from datetime import datetime

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
COMBINED_DIR = ROOT / 'ml' / 'datasets' / 'combined_dataset'
MODELS_DIR = ROOT / 'ml' / 'models'
MULTICLASS_DIR = MODELS_DIR / 'multiclass_candidates'
TRAINING_DIR = ROOT / 'ml' / 'training'


def get_dataset_stats():
    """Get dataset statistics for the training record."""
    stats = {}
    for split in ['train', 'val', 'test']:
        img_dir = COMBINED_DIR / split / 'images'
        lbl_dir = COMBINED_DIR / split / 'labels'
        if not img_dir.exists():
            continue
        images = list(img_dir.glob('*.jpg'))
        boxes = 0
        for lbl in lbl_dir.glob('*.txt'):
            text = lbl.read_text().strip()
            boxes += sum(1 for line in text.split('\n') if line.strip())
        stats[split] = {'images': len(images), 'annotations': boxes}
    return stats


def train_multiclass(epochs=40, imgsz=640, batch=8):
    """Train a multi-class YOLOv8n model."""
    
    print("=" * 70)
    print("MULTI-CLASS MICROPLASTIC MODEL TRAINING")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)
    
    # Verify dataset exists
    data_yaml = COMBINED_DIR / 'dataset.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    config = yaml.safe_load(data_yaml.read_text())
    print(f"\nDataset: {data_yaml}")
    print(f"Classes ({config['nc']}): {config['names']}")
    
    # Ensure absolute path in dataset.yaml
    config['path'] = str(COMBINED_DIR.resolve())
    data_yaml.write_text(yaml.safe_dump(config, default_flow_style=False))
    
    # Dataset stats
    ds_stats = get_dataset_stats()
    for split, s in ds_stats.items():
        print(f"  {split}: {s['images']} images, {s['annotations']} annotations")
    
    # Create output directories
    run_name = f"multiclass_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    MULTICLASS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 0
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU for training (this will be slower)")
    
    # Load pre-trained YOLOv8n
    base_model = ROOT / 'yolov8n.pt'
    if not base_model.exists():
        print("Base model yolov8n.pt not found, will be downloaded by ultralytics")
        model = YOLO('yolov8n.pt')
    else:
        model = YOLO(str(base_model))
    
    print(f"\nTraining configuration:")
    print(f"  Architecture: YOLOv8n")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Seed: 42")
    
    # Train
    start_time = time.time()
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=2,  # Conservative for Windows
        project=str(TRAINING_DIR),
        name=run_name,
        exist_ok=True,
        seed=42,
        deterministic=True,
        patience=15,
        plots=True,
        cache=False,
        close_mosaic=8,
        amp=False,
        verbose=True,
    )
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    
    # Validate model weights
    weights_dir = TRAINING_DIR / run_name / 'weights'
    for name in ['best.pt', 'last.pt']:
        source = weights_dir / name
        if not source.exists():
            print(f"  WARNING: {name} not found!")
            continue
        
        checkpoint = torch.load(source, weights_only=False, map_location='cpu')
        fitted = checkpoint.get('ema') or checkpoint.get('model')
        if fitted is not None:
            if not all(torch.isfinite(v).all() for v in fitted.state_dict().values()):
                raise RuntimeError(f'Non-finite weights in {name}! Model is corrupted.')
        
        # Copy to multiclass directory
        dst = MULTICLASS_DIR / name
        shutil.copy2(source, dst)
        print(f"  Saved: {dst} ({dst.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Copy dataset.yaml
    shutil.copy2(data_yaml, MULTICLASS_DIR / 'dataset.yaml')
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluate_multiclass(run_name)
    
    return run_name


def evaluate_multiclass(run_name=None):
    """Evaluate the multi-class model on val and test splits."""
    
    model_path = MULTICLASS_DIR / 'best.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    data_yaml = COMBINED_DIR / 'dataset.yaml'
    config = yaml.safe_load(data_yaml.read_text())
    config['path'] = str(COMBINED_DIR.resolve())
    data_yaml.write_text(yaml.safe_dump(config, default_flow_style=False))
    
    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    eval_dir = ROOT / 'ml' / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    output = {}
    for split in ['val', 'test']:
        print(f"\n  Evaluating on {split} split...")
        metric = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=640,
            batch=8,
            device=device,
            workers=0,
            project=str(eval_dir),
            name=f'multiclass_{split}',
            exist_ok=True,
            plots=True,
        )
        
        results = {k: float(v) for k, v in metric.results_dict.items()}
        
        if not all(math.isfinite(v) for v in results.values()):
            print(f"  WARNING: Non-finite metrics on {split}!")
        
        results['speed_ms_per_image'] = metric.speed
        output[split] = results
        
        # Print key metrics
        print(f"  Precision: {results.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {results.get('metrics/recall(B)', 0):.4f}")
        print(f"  mAP@50: {results.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@50-95: {results.get('metrics/mAP50-95(B)', 0):.4f}")
    
    # Save metrics
    metrics_path = eval_dir / 'multiclass_metrics.json'
    metrics_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Metrics saved: {metrics_path}")
    
    # Create model card
    model_card = {
        'model_name': 'YOLOv8n Multi-Class Microplastic Detector',
        'model_version': 'v1.0',
        'architecture': 'YOLOv8n (nano)',
        'framework': f'ultralytics + PyTorch {torch.__version__}',
        'training_date': datetime.now().isoformat(),
        'dataset_version': 'combined_v1',
        'classes': config['names'],
        'num_classes': config['nc'],
        'input_size': 640,
        'weight_file': 'best.pt',
        'weight_checksum': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'evaluation': output,
        'dataset_stats': get_dataset_stats(),
        'training_config': {
            'epochs': 40,
            'batch_size': 8,
            'image_size': 640,
            'seed': 42,
            'deterministic': True,
            'patience': 15,
            'close_mosaic': 8,
            'amp': False,
        },
        'limitations': [
            'Morphology detection only - does not identify polymer type.',
            'Performance may vary across microscope types and lighting conditions.',
            'Pixel measurements are not physical measurements without calibration.',
            'Synthetic training data supplements real microscopy images.',
            'DeepParticle images are cropped particles, not full microscopy fields.',
        ],
    }
    
    card_path = MULTICLASS_DIR / 'model_card.json'
    card_path.write_text(json.dumps(model_card, indent=2))
    print(f"  Model card saved: {card_path}")
    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-class microplastic detector')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    
    if args.evaluate_only:
        evaluate_multiclass()
    else:
        train_multiclass(args.epochs, args.imgsz, args.batch)
