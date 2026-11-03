from pathlib import Path
from collections import Counter
from threading import Lock
import io, json, time
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
MODEL_PATH=ROOT/'ml/models/best.pt'
DISCLAIMER='Image-based predictions identify suspected microplastic fragments. They do not chemically confirm plastic or polymer composition. Count bands are project-defined image indicators, not water-quality or safety standards.'

class Detector:
    def __init__(self):
        self.model=None; self.lock=Lock()
        torch.set_num_threads(4)
    def load(self):
        if not MODEL_PATH.exists(): raise RuntimeError('Trained model is missing. Restore ml/models/best.pt or run the training notebook.')
        if self.model is None: self.model=YOLO(str(MODEL_PATH))
        return self.model
    def info(self):
        model=self.load()
        metrics=ROOT/'ml/evaluation/metrics.json'
        return {'model':'YOLOv8n - trained fragment detector','classes':model.names,
                'metrics':json.loads(metrics.read_text()) if metrics.exists() else {},
                'dataset':'Microplastic-Fragments-WWTP-Dataset','disclaimer':DISCLAIMER,
                'assessment_thresholds':{'moderate':CONFIG['moderate_count'],'high':CONFIG['high_count']}}
    def predict(self, image, confidence, calibration):
        with self.lock:
            model=self.load(); start=time.perf_counter()
            # OpenCV converts validated RGB pixels into the BGR input expected by YOLO.
            result=model.predict(cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR),conf=confidence,imgsz=640,device='cpu',verbose=False)[0]
            elapsed=(time.perf_counter()-start)*1000
        rows=[]
        for i,box in enumerate(result.boxes):
            x1,y1,x2,y2=map(float,box.xyxy[0].tolist()); width=x2-x1; height=y2-y1
            row={'particle_id':i+1,'class_name':model.names[int(box.cls[0])],
                 'confidence':float(box.conf[0]),'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                 'width_pixels':width,'height_pixels':height,'area_pixels':width*height}
            if calibration is not None:
                row.update(width_um=width*calibration,height_um=height*calibration,area_um2=width*height*calibration**2)
            rows.append(row)
        count=len(rows)
        level='High' if count>=CONFIG['high_count'] else 'Moderate' if count>=CONFIG['moderate_count'] else 'Low'
        stats={'total_particles':count,'detections':rows,'class_distribution':dict(Counter(r['class_name'] for r in rows)),
               'average_confidence':sum(r['confidence'] for r in rows)/count if count else 0,
               'contamination_level':level,'inference_ms':round(elapsed,2),'confidence_threshold':confidence,
               'calibration_um_per_pixel':calibration,'image_width':image.width,'image_height':image.height,
               'disclaimer':DISCLAIMER}
        canvas=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR).copy()
        color=(100,155,25)
        for row in rows:
            x1,y1,x2,y2=(round(row[key]) for key in ['x1','y1','x2','y2'])
            cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
            label=f"{row['class_name']} {row['confidence']:.2f}"
            scale=min(.6,max(.25,image.width/1800))
            (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
            if tw>image.width-8:
                scale*=max(.1,(image.width-8)/tw)
                (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,scale,1)
            tx=max(0,min(x1,image.width-tw-6))
            ty=max(th+5,min(y1-5,image.height-baseline-2))
            cv2.rectangle(canvas,(tx,ty-th-4),(tx+tw+5,ty+baseline),color,-1)
            cv2.putText(canvas,label,(tx+2,ty-1),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)
        annotated=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        return stats,annotated

def decode_image(blob):
    try:
        with Image.open(io.BytesIO(blob)) as img:
            if img.format not in ('PNG','JPEG'): raise ValueError('Only JPEG and PNG images are supported.')
            if img.width*img.height>CONFIG['max_image_pixels']: raise ValueError('Image exceeds the 24 megapixel limit.')
            img.load(); return ImageOps.exif_transpose(img).convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        raise ValueError('The uploaded file is not a readable JPEG or PNG image.') from exc

def make_report(record, annotated_path, output_path):
    styles=getSampleStyleSheet(); story=[]
    def paragraph(text,style='Normal'): story.append(Paragraph(text,styles[style])); story.append(Spacer(1,8))
    paragraph('Microplastic Detection in Water','Title')
    paragraph('Microscopy analysis report - trained YOLOv8 fragment detector','Heading2')
    paragraph(f"Analysis: {record['created_at']} | ID: {record['id']}")
    paragraph(f"Input: {escape(record['input_name'])}")
    paragraph(f"Particles: {record['total_particles']} | Mean confidence: {record['average_confidence']:.1%} | Image count band: {record['contamination_level']}")
    paragraph(f"Class counts: {escape(str(record['class_distribution']))}")
    paragraph(f"Inference: {record['inference_ms']:.1f} ms | Threshold: {record['confidence_threshold']:.2f}")
    paragraph('Calibration: '+(f"{record['calibration_um_per_pixel']} micrometers per pixel" if record['calibration_um_per_pixel'] else 'Not provided; dimensions are in pixels.'))
    with Image.open(annotated_path) as img: w,h=img.size
    scale=min(490/w,260/h); story.append(PDFImage(str(annotated_path),width=w*scale,height=h*scale)); story.append(Spacer(1,12))
    paragraph(DISCLAIMER)
    paragraph(f"Count bands: Low < {CONFIG['moderate_count']}, Moderate {CONFIG['moderate_count']}-{CONFIG['high_count']-1}, High >= {CONFIG['high_count']}. These thresholds are configurable demonstrations.")
    paragraph('Detected particles','Heading2')
    physical=record['calibration_um_per_pixel'] is not None
    unit='um' if physical else 'px'
    data=[['ID','Dataset class','Confidence',f'Width ({unit})',f'Height ({unit})']]
    for row in record['detections']:
        data.append([str(row['particle_id']),Paragraph(escape(row['class_name'].replace('_',' ')),styles['Normal']),f"{row['confidence']:.1%}",f"{row['width_um' if physical else 'width_pixels']:.1f}",f"{row['height_um' if physical else 'height_pixels']:.1f}"])
    if not record['detections']: data.append(['-','No detections','-','-','-'])
    table=Table(data,colWidths=[35,155,95,100,100],repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#123e46')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf5f5'),colors.white])]))
    story.append(table)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawString(42,25,'Academic microscopy screening | dataset: andreusv / WWTP fragments, CC BY 4.0'); canvas.drawRightString(553,25,f'Page {doc.page}')
    SimpleDocTemplate(str(output_path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=36,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
