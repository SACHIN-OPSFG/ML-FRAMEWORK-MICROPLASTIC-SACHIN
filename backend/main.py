from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import json, sqlite3, re, math
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from .services import ROOT, CONFIG, Detector, decode_image, make_report

STORAGE=ROOT/'backend/storage'; STORAGE.mkdir(parents=True,exist_ok=True)
DB=STORAGE/'history.sqlite3'
with sqlite3.connect(DB) as conn: conn.execute('CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, created_at TEXT, payload TEXT)')
detector=Detector(); app=FastAPI(title='Microplastic Detection',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['*'])
@app.get('/results/{filename}')
def result_file(filename:str):
    if not re.fullmatch(r'[a-f0-9]{32}_(original|annotated)\.jpg',filename): raise HTTPException(404,'Result image not found.')
    path=STORAGE/filename
    if not path.exists(): raise HTTPException(404,'Result image not found.')
    return FileResponse(path,media_type='image/jpeg')

@app.get('/health')
def health():
    from .services import MODEL_PATH
    return {'status':'healthy' if MODEL_PATH.exists() else 'model_missing','model_ready':MODEL_PATH.exists()}

@app.get('/model-info')
def model_info():
    try: return detector.info()
    except RuntimeError as exc: raise HTTPException(503,str(exc))

@app.post('/predict')
async def predict(file:UploadFile=File(...),confidence:float=Form(0.25),calibration:float|None=Form(None)):
    if not math.isfinite(confidence) or not 0.01<=confidence<=1: raise HTTPException(400,'Confidence must be between 0.01 and 1.')
    if calibration is not None and (not math.isfinite(calibration) or calibration<=0): raise HTTPException(400,'Calibration must be a finite positive micrometers-per-pixel value.')
    if Path(file.filename or '').suffix.lower() not in {'.png','.jpg','.jpeg'}: raise HTTPException(400,'Choose a JPG, JPEG or PNG image.')
    blob=await file.read(CONFIG['max_upload_mb']*1024*1024+1)
    if len(blob)>CONFIG['max_upload_mb']*1024*1024: raise HTTPException(413,'Image exceeds the 15 MB upload limit.')
    try:
        image=decode_image(blob)
        record,annotated=await run_in_threadpool(detector.predict,image,confidence,calibration)
    except ValueError as exc: raise HTTPException(400,str(exc))
    except RuntimeError as exc: raise HTTPException(503,str(exc))
    ident=uuid4().hex; name=Path((file.filename or 'image').replace('\\','/')).name
    record.update(id=ident,created_at=datetime.now(timezone.utc).isoformat(),input_name=name,
                  result_image_url=f'/results/{ident}_annotated.jpg',original_image_url=f'/results/{ident}_original.jpg')
    image.save(STORAGE/f'{ident}_original.jpg',quality=95); annotated.save(STORAGE/f'{ident}_annotated.jpg',quality=95)
    with sqlite3.connect(DB) as conn: conn.execute('INSERT INTO analyses VALUES (?,?,?)',(ident,record['created_at'],json.dumps(record)))
    return record

@app.get('/history')
def history():
    with sqlite3.connect(DB) as conn: return [json.loads(row[0]) for row in conn.execute('SELECT payload FROM analyses ORDER BY created_at DESC LIMIT 100')]

class ReportRequest(BaseModel): analysis_id:str

@app.post('/generate-report')
def report(request:ReportRequest):
    if not re.fullmatch('[a-f0-9]{32}',request.analysis_id): raise HTTPException(400,'Invalid analysis ID.')
    with sqlite3.connect(DB) as conn: row=conn.execute('SELECT payload FROM analyses WHERE id=?',(request.analysis_id,)).fetchone()
    if row is None: raise HTTPException(404,'Analysis was not found.')
    record=json.loads(row[0]); output=STORAGE/f'{request.analysis_id}_report.pdf'
    make_report(record,STORAGE/f'{request.analysis_id}_annotated.jpg',output)
    return FileResponse(output,media_type='application/pdf',filename='Microplastic_Analysis_Report.pdf')

@app.get('/samples')
def samples(): return [p.name for p in (ROOT/'sample_data').glob('*.jpg')]

if (ROOT/'sample_data').exists(): app.mount('/sample-data',StaticFiles(directory=ROOT/'sample_data'),name='samples')
if (ROOT/'frontend/dist').exists(): app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='dashboard')
