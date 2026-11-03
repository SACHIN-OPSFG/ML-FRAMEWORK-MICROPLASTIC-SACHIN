from pathlib import Path
from collections import Counter
from threading import Lock
import io, json, time, csv, hashlib
import cv2, numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'backend/config.json').read_text())

# Model registry: maps model_id to weight path and metadata
MODEL_REGISTRY = {
    'baseline': {
        'name': 'YOLOv8n - Baseline Fragment Detector',
        'path': ROOT/'ml/models/best.pt',
        'backup_path': ROOT/'ml/models/baseline_single_class/best.pt',
        'metrics_path': ROOT/'ml/evaluation/metrics.json',
        'dataset': 'Microplastic-Fragments-WWTP-Dataset',
        'version': '1.0',
        'type': 'single_class',
    },
    'multiclass': {
        'name': 'YOLOv8n - Multi-Class Morphology Detector',
        'path': ROOT/'ml/models/multiclass_candidates/best.pt',
        'metrics_path': ROOT/'ml/evaluation/multiclass_metrics.json',
        'dataset': 'Combined Multi-Class Dataset (WWTP + DeepParticle + Synthetic)',
        'version': '1.0',
        'type': 'multi_class',
    },
}

# Class colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    'fiber': (255, 140, 50),        # Blue
    'fragment': (100, 155, 25),     # Green (original color)
    'film': (50, 200, 255),         # Yellow
    'pellet': (200, 50, 200),       # Purple
    'foam': (50, 255, 200),         # Cyan
    'bead': (50, 50, 255),          # Red
    'microplastic_fragment': (100, 155, 25),  # Green (baseline)
}

DISCLAIMER='Results are image-based detections of suspected particles. They do not independently confirm that a particle is plastic or identify its polymer composition. Confirmatory chemical analysis and a validated sampling protocol may be required for scientific conclusions.'

# Keep backward-compatible MODEL_PATH for existing code
MODEL_PATH=ROOT/'ml/models/best.pt'


class Detector:
    def __init__(self):
        self._models = {}  # Cache: model_id -> YOLO model
        self.lock = Lock()
        torch.set_num_threads(4)

    def _get_model_path(self, model_id='baseline'):
        """Get the weight file path for a model ID."""
        info = MODEL_REGISTRY.get(model_id)
        if info is None:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}")
        path = info['path']
        if not path.exists() and 'backup_path' in info:
            path = info['backup_path']
        return path

    def load(self, model_id='baseline'):
        """Load and cache a model by ID."""
        path = self._get_model_path(model_id)
        if not path.exists():
            raise RuntimeError(f"Model weights not found: {path}. Train the model or select a different model.")
        if model_id not in self._models:
            self._models[model_id] = YOLO(str(path))
        return self._models[model_id]

    def available_models(self):
        """List all available models."""
        models = []
        for mid, info in MODEL_REGISTRY.items():
            path = info['path']
            if not path.exists() and 'backup_path' in info:
                path = info['backup_path']
            available = path.exists()
            entry = {
                'model_id': mid,
                'name': info['name'],
                'version': info['version'],
                'type': info['type'],
                'available': available,
                'dataset': info['dataset'],
            }
            if available:
                try:
                    model = self.load(mid)
                    entry['classes'] = model.names
                    entry['num_classes'] = len(model.names)
                except Exception:
                    entry['classes'] = {}
                    entry['num_classes'] = 0
            models.append(entry)
        return models

    def info(self, model_id='baseline'):
        """Get model info including evaluation metrics."""
        model = self.load(model_id)
        reg = MODEL_REGISTRY.get(model_id, MODEL_REGISTRY['baseline'])
        metrics_path = reg.get('metrics_path', ROOT/'ml/evaluation/metrics.json')
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        # Also try model card
        model_card = {}
        card_path = reg['path'].parent / 'model_card.json'
        if card_path.exists():
            model_card = json.loads(card_path.read_text())

        return {
            'model_id': model_id,
            'model': reg['name'],
            'version': reg['version'],
            'type': reg['type'],
            'classes': model.names,
            'num_classes': len(model.names),
            'metrics': metrics,
            'dataset': reg['dataset'],
            'disclaimer': DISCLAIMER,
            'assessment_thresholds': {'moderate': CONFIG['moderate_count'], 'high': CONFIG['high_count']},
            'model_card': model_card,
        }

    def predict(self, image, confidence, calibration, model_id='baseline'):
        """Run detection with the specified model."""
        with self.lock:
            model = self.load(model_id)
            start = time.perf_counter()
            result = model.predict(
                cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                conf=confidence, imgsz=640, device='cpu', verbose=False
            )[0]
            elapsed = (time.perf_counter() - start) * 1000

        rows = []
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            width = x2 - x1
            height = y2 - y1
            class_name = model.names[int(box.cls[0])]
            
            # For multi-class model: refine morphology using geometric research criteria
            if model_id == 'multiclass' and len(model.names) > 1:
                aspect = max(width, height) / max(min(width, height), 1.0)
                area = width * height
                if aspect >= 2.4:
                    class_name = 'fiber'
                elif 0.82 <= width / max(height, 1.0) <= 1.22 and area < 4000:
                    class_name = 'bead' if area < 1500 else 'pellet'
                elif aspect >= 1.7:
                    class_name = 'film'
                elif area > 12000:
                    class_name = 'foam'
                else:
                    class_name = 'fragment'

            row = {
                'particle_id': i + 1,
                'class_name': class_name,
                'confidence': float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width_pixels': width,
                'height_pixels': height,
                'area_pixels': width * height,
            }
            if calibration is not None:
                row.update(
                    width_um=width * calibration,
                    height_um=height * calibration,
                    area_um2=width * height * calibration ** 2,
                )
            rows.append(row)

        count = len(rows)
        level = 'High' if count >= CONFIG['high_count'] else 'Moderate' if count >= CONFIG['moderate_count'] else 'Low'

        stats = {
            'total_particles': count,
            'detections': rows,
            'class_distribution': dict(Counter(r['class_name'] for r in rows)),
            'average_confidence': sum(r['confidence'] for r in rows) / count if count else 0,
            'contamination_level': level,
            'inference_ms': round(elapsed, 2),
            'confidence_threshold': confidence,
            'calibration_um_per_pixel': calibration,
            'image_width': image.width,
            'image_height': image.height,
            'model_id': model_id,
            'model_name': MODEL_REGISTRY.get(model_id, {}).get('name', 'Unknown'),
            'disclaimer': DISCLAIMER,
        }

        # Draw annotated image with class-specific colors
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
        for row in rows:
            x1, y1, x2, y2 = (round(row[key]) for key in ['x1', 'y1', 'x2', 'y2'])
            color = CLASS_COLORS.get(row['class_name'], (100, 155, 25))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{row['class_name']} {row['confidence']:.2f}"
            scale = min(.6, max(.25, image.width / 1800))
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw > image.width - 8:
                scale *= max(.1, (image.width - 8) / tw)
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            tx = max(0, min(x1, image.width - tw - 6))
            ty = max(th + 5, min(y1 - 5, image.height - baseline - 2))
            cv2.rectangle(canvas, (tx, ty - th - 4), (tx + tw + 5, ty + baseline), color, -1)
            cv2.putText(canvas, label, (tx + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

        annotated = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return stats, annotated


def decode_image(blob):
    try:
        with Image.open(io.BytesIO(blob)) as img:
            if img.format not in ('PNG','JPEG'): raise ValueError('Only JPEG and PNG images are supported.')
            if img.width*img.height>CONFIG['max_image_pixels']: raise ValueError('Image exceeds the 24 megapixel limit.')
            img.load(); return ImageOps.exif_transpose(img).convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('The uploaded file is not a readable JPEG or PNG image.') from exc


def get_dataset_summary():
    """Get dataset statistics for the dashboard."""
    summary = {
        'sources': [],
        'total_original_images': 0,
        'total_synthetic_images': 0,
        'total_augmented_images': 0,
        'total_annotated_objects': 0,
        'num_sources': 0,
        'num_classes': 6,
        'class_names': ['fiber', 'fragment', 'film', 'pellet', 'foam', 'bead'],
        'splits': {},
        'class_distribution': {},
        'licence_status': 'All datasets CC BY 4.0, Apache 2.0, or CC0 1.0',
    }

    # Read manifests if available
    manifests_dir = ROOT / 'ml' / 'datasets' / 'manifests'
    
    inventory_path = manifests_dir / 'dataset_inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        for ds in inventory.get('datasets', []):
            source = {
                'name': ds.get('source_name', ''),
                'license': ds.get('license', ''),
                'images': ds.get('total_images', 0),
                'annotations': ds.get('total_annotations', 0),
                'type': ds.get('image_type', ''),
            }
            summary['sources'].append(source)
            summary['num_sources'] += 1
            
            if ds.get('image_type') == 'original_microscopy':
                summary['total_original_images'] += ds.get('total_images', 0)
            elif ds.get('image_type') == 'synthetic_microscopy':
                summary['total_synthetic_images'] += ds.get('total_images', 0)

    stats_path = manifests_dir / 'combined_dataset_stats.json'
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for split, split_data in stats.items():
            summary['splits'][split] = split_data
            summary['total_annotated_objects'] += split_data.get('annotations', 0)
            for cls, cnt in split_data.get('class_distribution', {}).items():
                summary['class_distribution'][cls] = summary['class_distribution'].get(cls, 0) + cnt

    # Count actual combined dataset files
    combined_dir = ROOT / 'ml' / 'datasets' / 'combined_dataset'
    for split in ['train', 'val', 'test']:
        img_dir = combined_dir / split / 'images'
        if img_dir.exists():
            imgs = list(img_dir.glob('*.jpg'))
            if split not in summary['splits']:
                summary['splits'][split] = {'images': len(imgs)}

    return summary


def make_csv_export(record):
    """Export detection data as CSV string."""
    output = io.StringIO()
    fieldnames = [
        'particle_id', 'class_name', 'confidence',
        'x1', 'y1', 'x2', 'y2',
        'width_pixels', 'height_pixels', 'area_pixels',
    ]
    has_cal = record.get('calibration_um_per_pixel') is not None
    if has_cal:
        fieldnames.extend(['width_um', 'height_um', 'area_um2'])

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    for det in record.get('detections', []):
        writer.writerow(det)

    return output.getvalue()


def make_report(record, annotated_path, output_path):
    styles=getSampleStyleSheet(); story=[]
    def paragraph(text,style='Normal'): story.append(Paragraph(text,styles[style])); story.append(Spacer(1,8))
    paragraph('Microplastic Detection in Water','Title')

    model_name = record.get('model_name', 'YOLOv8 detector')
    paragraph(f'Microscopy analysis report - {escape(model_name)}','Heading2')
    paragraph(f"Analysis: {record['created_at']} | ID: {record['id']}")
    paragraph(f"Input: {escape(record['input_name'])}")
    paragraph(f"Model: {escape(model_name)} | Version: {record.get('model_id', 'baseline')}")
    paragraph(f"Particles: {record['total_particles']} | Mean confidence: {record['average_confidence']:.1%} | Image count band: {record['contamination_level']}")

    # Class-wise counts
    class_dist = record.get('class_distribution', {})
    if class_dist:
        counts_str = ', '.join(f"{k}: {v}" for k, v in sorted(class_dist.items(), key=lambda x: -x[1]))
        paragraph(f"Class counts: {escape(counts_str)}")

    paragraph(f"Inference: {record['inference_ms']:.1f} ms | Threshold: {record['confidence_threshold']:.2f}")
    paragraph('Calibration: '+(f"{record['calibration_um_per_pixel']} micrometers per pixel" if record['calibration_um_per_pixel'] else 'Not provided; dimensions are in pixels.'))

    with Image.open(annotated_path) as img: w,h=img.size
    scale=min(490/w,260/h); story.append(PDFImage(str(annotated_path),width=w*scale,height=h*scale)); story.append(Spacer(1,12))
    paragraph(DISCLAIMER)
    paragraph(f"Count bands: Low < {CONFIG['moderate_count']}, Moderate {CONFIG['moderate_count']}-{CONFIG['high_count']-1}, High >= {CONFIG['high_count']}. These thresholds are configurable demonstrations.")

    # Scientific limitations section
    paragraph('Scientific Limitations','Heading2')
    limitations = [
        'Morphology detection is not polymer identification.',
        'Visible-light images may contain sediment, bubbles, and natural fibers.',
        'Performance may vary between microscopes, lighting, and sample matrices.',
        'Pixel measurements are not physical measurements without calibration.',
        'Particle count per image is not concentration per litre.',
    ]
    for lim in limitations:
        paragraph(f"• {lim}")

    paragraph('Detected particles','Heading2')
    physical=record['calibration_um_per_pixel'] is not None
    unit='µm' if physical else 'px'
    data=[['ID','Morphology Class','Confidence',f'Width ({unit})',f'Height ({unit})',f'Area ({unit}²)']]
    for row in record['detections']:
        w_val = row.get('width_um' if physical else 'width_pixels', 0)
        h_val = row.get('height_um' if physical else 'height_pixels', 0)
        a_val = row.get('area_um2' if physical else 'area_pixels', 0)
        data.append([
            str(row['particle_id']),
            Paragraph(escape(row['class_name'].replace('_',' ')), styles['Normal']),
            f"{row['confidence']:.1%}",
            f"{w_val:.1f}",
            f"{h_val:.1f}",
            f"{a_val:.1f}",
        ])
    if not record['detections']: data.append(['-','No detections','-','-','-','-'])
    table=Table(data,colWidths=[30,120,75,80,80,85],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#123e46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf5f5'),colors.white])]))
    story.append(table)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8)
        canvas.drawString(42,25,'Academic microscopy screening | Multi-class morphology detection')
        canvas.drawRightString(553,25,f'Page {doc.page}')
    SimpleDocTemplate(str(output_path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
