"""Real-model API integration checks and sample artifacts; run from project root."""
from pathlib import Path
import io,json,sys
from PIL import Image
from fastapi.testclient import TestClient
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.main import app
from backend import services

def verify():
    client=TestClient(app); checks=[]; output=ROOT/'output/verification';output.mkdir(parents=True,exist_ok=True)
    def check(name, condition):
        assert condition,name
        checks.append({'test':name,'passed':True});print('PASS',name)

    # ===== Original 24 checks =====
    check('Backend healthy with trained model',client.get('/health').json()['model_ready'])
    check('Model exposes actual evaluation metrics',bool(client.get('/model-info').json()['metrics']['test']))
    image=ROOT/'sample_data/test_sample_01.jpg'; blob=image.read_bytes()
    response=client.post('/predict',files={'file':(image.name,blob,'image/jpeg')},data={'confidence':'.25'})
    check('Real held-out image accepted',response.status_code==200)
    record=response.json(); count=record['total_particles']
    check('Particle counts agree with records and classes',count==len(record['detections'])==sum(record['class_distribution'].values()))
    check('Every detection has valid bounds and confidence',all(0<=d['x1']<d['x2']<=record['image_width']+.01 and 0<=d['y1']<d['y2']<=record['image_height']+.01 and .25<=d['confidence']<=1 for d in record['detections']))
    calibration_record=client.post('/predict',files={'file':(image.name,blob,'image/jpeg')},data={'confidence':'.25','calibration':'.5'}).json()
    check('Calibration measurements agree',all(abs(d['width_um']-d['width_pixels']*.5)<1e-5 and abs(d['area_um2']-d['area_pixels']*.25)<1e-5 for d in calibration_record['detections']))
    annotated=client.get(record['result_image_url'])
    check('Annotated image generated',annotated.status_code==200 and Image.open(io.BytesIO(annotated.content)).size==(record['image_width'],record['image_height']))
    (output/'sample_annotated.jpg').write_bytes(annotated.content)
    report=client.post('/generate-report',json={'analysis_id':record['id']})
    check('PDF generated from saved analysis',report.status_code==200 and report.content.startswith(b'%PDF'))
    (output/'sample_analysis_report.pdf').write_bytes(report.content)
    (output/'sample_analysis.json').write_text(json.dumps(record,indent=2))
    check('Analysis persisted in history',any(h['id']==record['id'] for h in client.get('/history').json()))
    no_cal=client.post('/predict',files={'file':('sample.jpg',blob)},data={'confidence':'.25'}).json()
    check('No physical dimensions without calibration',all('width_um' not in row for row in no_cal['detections']))
    check('Unsupported file rejected',client.post('/predict',files={'file':('bad.txt',b'hello')}).status_code==400)
    check('Corrupt image rejected',client.post('/predict',files={'file':('bad.png',b'not an image')}).status_code==400)
    check('Disguised unsupported image rejected',client.post('/predict',files={'file':('bad.jpg',b'GIF89a')}).status_code==400)
    check('Negative calibration rejected',client.post('/predict',files={'file':('sample.jpg',blob)},data={'calibration':-1}).status_code==400)
    check('Nonfinite calibration rejected',client.post('/predict',files={'file':('sample.jpg',blob)},data={'calibration':'nan'}).status_code==400)
    check('Invalid confidence rejected',client.post('/predict',files={'file':('sample.jpg',blob)},data={'confidence':1.1}).status_code==400)
    check('Oversized upload rejected',client.post('/predict',files={'file':('large.jpg',b'0'*(15*1024*1024+1))}).status_code==413)
    buff=io.BytesIO();Image.new('RGB',(5000,5000),'white').save(buff,format='PNG')
    check('Oversized decoded image rejected',client.post('/predict',files={'file':('large.png',buff.getvalue())}).status_code==400)
    buff=io.BytesIO();Image.new('RGB',(640,640),'white').save(buff,format='PNG')
    zero=client.post('/predict',files={'file':('blank.png',buff.getvalue())},data={'confidence':1}).json()
    check('Zero detections handled safely',zero['total_particles']==0 and zero['average_confidence']==0)
    check('Unknown report returns 404',client.post('/generate-report',json={'analysis_id':'0'*32}).status_code==404)
    check('Report ID cannot escape storage',client.post('/generate-report',json={'analysis_id':'../../secrets'}).status_code==400)
    check('Internal database is not downloadable',client.get('/results/history.sqlite3').status_code==404)
    cors=client.options('/predict',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    check('Configured frontend origin allowed',cors.headers.get('access-control-allow-origin')=='http://localhost:5173')
    missing=services.Detector()
    try: missing.load('nonexistent_model_xyz'); failure=False
    except (RuntimeError, ValueError) as e: failure='missing' in str(e).lower() or 'unknown' in str(e).lower() or 'not found' in str(e).lower()
    check('Missing model gives readable configuration error',failure)

    # ===== New Phase 2 checks =====
    # Models listing endpoint
    models_resp=client.get('/models')
    check('Models endpoint returns list',models_resp.status_code==200 and isinstance(models_resp.json(),list))
    models_list=models_resp.json()
    check('Baseline model listed and available',any(m['model_id']=='baseline' and m['available'] for m in models_list))
    check('Multiclass model listed in registry',any(m['model_id']=='multiclass' for m in models_list))

    # Model selection in predict
    baseline_resp=client.post('/predict',files={'file':(image.name,blob,'image/jpeg')},data={'confidence':'.25','model_id':'baseline'})
    check('Predict with model_id=baseline works',baseline_resp.status_code==200)
    baseline_rec=baseline_resp.json()
    check('Baseline response includes model_id',baseline_rec.get('model_id')=='baseline')
    check('Baseline response includes model_name',bool(baseline_rec.get('model_name')))
    check('Baseline response includes disclaimer',bool(baseline_rec.get('disclaimer')))
    check('Baseline response includes class_distribution',isinstance(baseline_rec.get('class_distribution'),dict))

    # CSV export
    csv_resp=client.get(f'/exports/{baseline_rec["id"]}.csv')
    check('CSV export returns 200',csv_resp.status_code==200)
    check('CSV has content-type text/csv','text/csv' in csv_resp.headers.get('content-type',''))
    csv_text=csv_resp.text
    check('CSV has header row','particle_id' in csv_text and 'class_name' in csv_text)

    # Dataset summary
    ds_resp=client.get('/dataset/summary')
    check('Dataset summary returns 200',ds_resp.status_code==200)
    ds=ds_resp.json()
    check('Dataset summary has class_names','class_names' in ds and len(ds['class_names'])>=1)
    check('Dataset summary has sources',isinstance(ds.get('sources'),list))

    # Evaluation endpoint
    eval_resp=client.get('/evaluation/baseline')
    check('Evaluation endpoint returns 200',eval_resp.status_code==200)
    eval_data=eval_resp.json()
    check('Evaluation returns model info','model_id' in eval_data)

    # Individual history record
    hist_resp=client.get(f'/history/{baseline_rec["id"]}')
    check('Individual history record works',hist_resp.status_code==200 and hist_resp.json()['id']==baseline_rec['id'])

    # Invalid model_id
    bad_model=client.post('/predict',files={'file':(image.name,blob,'image/jpeg')},data={'confidence':'.25','model_id':'nonexistent'})
    check('Invalid model_id rejected',bad_model.status_code in (400,503))

    summary={'checks':checks,'passed':len(checks),'sample_particles':count,'sample_analysis_id':record['id']}
    (output/'test_results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':verify()
