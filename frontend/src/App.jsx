import React,{useEffect,useState,useRef,useCallback} from 'react';
import {Waves,Microscope,Upload,ArrowUpRight,FlaskConical,Clock,Download,ScanLine,CheckCircle2,AlertCircle,Layers,Target,Activity,ChevronRight,Database,BarChart3,PieChart as PieChartIcon,FileText,Info} from 'lucide-react';
import {ResponsiveContainer,BarChart,Bar,XAxis,YAxis,Tooltip,CartesianGrid,PieChart,Pie,Cell,Legend} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '';
const fullUrl = u => !u ? '' : (u.startsWith('http') || u.startsWith('blob:')) ? u : (API_BASE + u);
async function api(url,options){const r=await fetch(fullUrl(url),options);if(!r.ok){let message;try{const j=await r.json();message=typeof j.detail==='string'?j.detail:'Check the supplied values.'}catch{message='The server could not process this request.'}throw Error(message)}return r}
const pct=n=>(n*100).toFixed(1)+'%';

/* Class colors matching backend CLASS_COLORS (converted to hex for frontend) */
const CLASS_COLORS={
  fiber:'#328cff',
  fragment:'#199b19',
  film:'#ffc832',
  pellet:'#c832c8',
  foam:'#32ffc8',
  bead:'#ff3232',
  microplastic_fragment:'#199b19',
};
const PIE_COLORS=['#328cff','#199b19','#ffc832','#c832c8','#32ffc8','#ff3232','#888888'];

export default function App(){
 const [tab,setTab]=useState('analyze'),[models,setModels]=useState([]),[selectedModel,setSelectedModel]=useState('baseline'),[info,setInfo]=useState(null),[history,setHistory]=useState([]),[samples,setSamples]=useState([]),[file,setFile]=useState(null),[preview,setPreview]=useState(''),[result,setResult]=useState(null),[busy,setBusy]=useState(false),[reportBusy,setReportBusy]=useState(false),[csvBusy,setCsvBusy]=useState(false),[error,setError]=useState(''),[connectionError,setConnectionError]=useState(''),[conf,setConf]=useState(.25),[calibration,setCalibration]=useState(''),[drag,setDrag]=useState(false),[datasetSummary,setDatasetSummary]=useState(null),[evalModelId,setEvalModelId]=useState('baseline');const input=useRef();
 const loadHistory=()=>api('/history').then(r=>r.json()).then(setHistory);
 const loadModels=useCallback(()=>api('/models').then(r=>r.json()).then(m=>{setModels(m);return m}),[]);
 const loadInfo=useCallback((mid)=>api('/model-info?model_id='+encodeURIComponent(mid)).then(r=>r.json()),[]);

 useEffect(()=>{Promise.all([
   loadModels(),
   loadInfo('baseline').then(setInfo),
   loadHistory(),
   api('/samples').then(r=>r.json()).then(setSamples),
 ]).catch(e=>setConnectionError('Backend unavailable: '+e.message))},[]);

 /* Load dataset summary when switching to dataset tab */
 useEffect(()=>{if(tab==='dataset'&&!datasetSummary){api('/dataset/summary').then(r=>r.json()).then(setDatasetSummary).catch(()=>{})}},[tab]);

 /* When model selection changes, update info */
 useEffect(()=>{loadInfo(selectedModel).then(setInfo).catch(()=>{})},[selectedModel]);

 useEffect(()=>{if(!file){setPreview('');return}const u=URL.createObjectURL(file);setPreview(u);return()=>URL.revokeObjectURL(u)},[file]);
 function choose(f){if(!f)return;setError('');if(!/\.(png|jpe?g)$/i.test(f.name)||!['image/png','image/jpeg'].includes(f.type)){setError('Choose a JPG or PNG image.');return}if(f.size>15*1024*1024){setError('Image must be smaller than 15 MB.');return}setFile(f);setResult(null)}
 async function sample(){try{const response=await api('/sample-data/'+samples[0]);choose(new File([await response.blob()],samples[0],{type:'image/jpeg'}))}catch(e){setError(e.message)}}
 async function analyze(){setBusy(true);setError('');try{const data=new FormData();data.append('file',file);data.append('confidence',conf);data.append('model_id',selectedModel);if(calibration!=='')data.append('calibration',calibration);const r=await api('/predict',{method:'POST',body:data});setResult(await r.json());await loadHistory()}catch(e){setError(e.message)}finally{setBusy(false)}}
 async function report(){setReportBusy(true);try{const r=await api('/generate-report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({analysis_id:result.id})});const u=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=u;a.download='Microplastic_Analysis_Report.pdf';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}catch(e){setError(e.message)}finally{setReportBusy(false)}}
 async function exportCsv(){setCsvBusy(true);try{const r=await api('/exports/'+result.id+'.csv');const u=URL.createObjectURL(await r.blob());const a=document.createElement('a');a.href=u;a.download='detections_'+result.id.substring(0,8)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}catch(e){setError(e.message)}finally{setCsvBusy(false)}}

 const chart=result?Object.entries(result.class_distribution).map(([name,count])=>({name:name.replaceAll('_',' '),count})):[];
 const pieData=chart.map((d,i)=>({...d,fill:PIE_COLORS[i%PIE_COLORS.length]}));

 /* Class summary cards for multi-class results */
 const classCards=result?Object.entries(result.class_distribution).sort((a,b)=>b[1]-a[1]):[];

 const navItems=[
   ['analyze','Sample analysis',ScanLine],
   ['history','Analysis history',Clock],
   ['model','Model & evaluation',Layers],
   ['dataset','Dataset overview',Database],
 ];
 const tabTitles={analyze:'Sample analysis',history:'Analysis history',model:'Model & evaluation',dataset:'Dataset overview'};
 const tabHeadings={analyze:'A closer look at your sample.',history:'Your analysis archive.',model:'Evidence behind the model.',dataset:'What the model learned from.'};
 const tabSubtitles={analyze:'Detect suspected microplastic particles, explore measurements, and document your findings.',history:'Revisit saved samples and generate reports from previous analyses.',model:'Actual training and held-out evaluation results. No estimated accuracy.',dataset:'Source datasets, class distribution, and split statistics.'};

 return <div className="shell"><aside><a className="brand" href="/"><span><Waves size={25}/></span><div>MICROPLASTIC<small>MICROSCOPY LAB</small></div></a><div className="workspace">RESEARCH WORKSPACE</div><nav>{navItems.map(([id,label,Icon])=><button key={id} className={tab===id?'active':''} onClick={()=>setTab(id)}><Icon size={19}/>{label}{tab===id&&<ChevronRight size={15}/>}</button>)}</nav><div className="side-note"><FlaskConical size={23}/><b>From pixels to insight.</b><p>A reproducible workflow for screening microscopic water-sample images.</p><span>ACADEMIC RESEARCH PROJECT</span></div><div className="side-bottom"><i className={info?'dot':'dot off'}/>{info?'Trained model connected':'Connecting to model…'}<small>YOLOv8 · Computer vision</small></div></aside>
 <main><header><span>Workspace / <b>{tabTitles[tab]||'Dashboard'}</b></span><div className="header-tag"><Microscope size={16}/> IMAGE-BASED SCREENING</div></header><div className="content"><div className="title-row"><div><div className="eyebrow">WATER QUALITY RESEARCH</div><h1>{tabHeadings[tab]||''}</h1><p className="subtitle">{tabSubtitles[tab]||''}</p></div><span className="version">PHASE 02 <ArrowUpRight size={15}/></span></div>
 {(error||connectionError)&&<div role="alert" className="error"><AlertCircle size={18}/>{error||connectionError}</div>}

 {/* ===== ANALYZE TAB ===== */}
 {tab==='analyze'&&<><div className="analysis-grid"><section className="panel upload-panel"><div className="panel-title"><span className="step">01</span><div><h2>Prepare your sample</h2><p>Upload a microscopic image for analysis</p></div></div>

 {/* Model selector */}
 <div className="model-selector">
  <label>Detection model
   <select value={selectedModel} onChange={e=>{setSelectedModel(e.target.value);setResult(null)}}>
    {models.filter(m=>m.available).map(m=><option key={m.model_id} value={m.model_id}>{m.name} ({m.num_classes} class{m.num_classes!==1?'es':''})</option>)}
    {models.filter(m=>!m.available).map(m=><option key={m.model_id} value={m.model_id} disabled>{m.name} (not available)</option>)}
   </select>
  </label>
  {info&&<small className="model-hint">{info.type==='multi_class'?`Multi-class: ${Object.values(info.classes).join(', ')}`:`Single class: ${Object.values(info.classes).join(', ')}`}</small>}
 </div>

 <div className={'dropzone '+(drag?'drag':'')} role="button" tabIndex={0} aria-label="Upload microscope image" onClick={()=>input.current.click()} onKeyDown={e=>{if(e.key==='Enter'||e.key===' ')input.current.click()}} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);choose(e.dataTransfer.files[0])}}>{(preview||result?.original_image_url)?<><img src={preview||fullUrl(result?.original_image_url)} alt="Selected microscopic sample"/><div className="image-label">{file?.name||result?.input_name||"Sample"} · Click to change</div></>:<><div className="upload-icon"><Upload size={28}/></div><b>Drop your image here</b><p>or <em>browse files</em> from your computer</p><small>JPG or PNG · Up to 15 MB · 24 megapixels</small></>}<input ref={input} type="file" accept="image/png,image/jpeg" hidden onChange={e=>choose(e.target.files[0])}/></div><div className="sample-link"><span>Just exploring?</span><button onClick={sample} disabled={!samples.length||busy}>Try a real test sample <ArrowUpRight size={14}/></button></div><div className="controls"><label>Confidence threshold <b>{pct(conf)}</b><input aria-label="Confidence threshold" type="range" min="0.05" max="0.95" step="0.05" value={conf} onChange={e=>setConf(Number(e.target.value))}/></label><label>Microscope calibration <span className="optional">OPTIONAL</span><div className="calibration"><input aria-label="Microscope calibration" type="number" min="0.000001" step="any" placeholder="e.g. 0.50" value={calibration} onChange={e=>setCalibration(e.target.value)}/><span>µm / pixel</span></div><small>Without calibration, dimensions are reported in pixels.</small></label></div><button className="primary analyze" disabled={!file||busy||!info} onClick={analyze}>{busy ? (<><span className="spinner"/>Analyzing your sample…</>) : (<><ScanLine size={18}/>Analyze sample<ArrowUpRight size={18}/></>)}</button></section>

 <section className="panel result-panel"><div className="panel-title"><span className="step">02</span><div><h2>Detection preview</h2><p>Bounding boxes from your trained YOLOv8 model</p></div>{result&&<span className="pill"><CheckCircle2 size={13}/>Complete</span>}</div>{result?<div className="annotated"><img src={fullUrl(result.result_image_url)} alt="Annotated microplastic detections"/><div className="preview-bottom"><span>{result.input_name}</span><span>{result.model_name||'Baseline'}</span><span>{result.inference_ms.toFixed(0)} ms inference</span></div></div>:<div className="empty-preview"><div className="lens"><ScanLine size={65}/><span/></div><h3>Your sample, examined.</h3><p>Upload an image and run analysis to see<br/>detected particles and their bounding boxes.</p><div className="preview-chips"><span>Detect</span><i/><span>Classify</span><i/><span>Measure</span><i/><span>Report</span></div></div>}<div className="scientific-note"><FlaskConical size={18}/><p><b>Visual screening, interpreted carefully.</b> Predictions indicate suspected particles. Chemical testing is required to confirm polymer composition.</p></div></section></div>

 {/* Stats row */}
 <div className="stats">{[[Activity,'Suspected particles',result?.total_particles??'—','Objects above your confidence threshold'],[Target,'Mean confidence',result?pct(result.average_confidence):'—','Model confidence, not accuracy'],[Waves,'Image count band',result?.contamination_level??'—','Project-defined demonstration thresholds'],[Layers,'Detected classes',result?Object.keys(result.class_distribution).length:'—','Morphology classes found']].map(([Icon,label,value,note])=><div className="stat" key={label}><div><span>{label}</span><Icon size={18}/></div><strong>{value}</strong><small>{note}</small></div>)}</div>

 {/* Class distribution cards */}
 {result&&classCards.length>0&&<div className="class-cards">{classCards.map(([cls,cnt])=><div className="class-card" key={cls}><div className="class-dot" style={{background:CLASS_COLORS[cls]||'#888'}}/><span className="class-name">{cls.replaceAll('_',' ')}</span><strong>{cnt}</strong></div>)}</div>}

 {result&&<><div className="detail-grid"><section className="panel"><h2>Particle distribution</h2><p className="muted">Count by morphology class</p>{chart.length?<ResponsiveContainer width="100%" height={190}><BarChart data={chart} margin={{left:0,right:24,bottom:0}}><CartesianGrid strokeDasharray="4 4" vertical={false}/><XAxis dataKey="name" tick={{fontSize:12}}/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="count" radius={[6,6,0,0]} maxBarSize={90}>{chart.map((entry,i)=><Cell key={i} fill={CLASS_COLORS[entry.name.replaceAll(' ','_')]||PIE_COLORS[i%PIE_COLORS.length]}/>)}</Bar></BarChart></ResponsiveContainer>:<p>No particles above the selected threshold.</p>}</section>

 {/* Pie chart for class distribution */}
 {chart.length>1&&<section className="panel"><h2>Class proportions</h2><p className="muted">Relative distribution of detected classes</p><ResponsiveContainer width="100%" height={190}><PieChart><Pie data={pieData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({name,percent})=>`${name} ${(percent*100).toFixed(0)}%`}>{pieData.map((entry,i)=><Cell key={i} fill={CLASS_COLORS[entry.name.replaceAll(' ','_')]||PIE_COLORS[i%PIE_COLORS.length]}/>)}</Pie><Tooltip/></PieChart></ResponsiveContainer></section>}

 {chart.length<=1&&<section className="panel report-card"><div className="report-icon"><Download/></div><h2>Keep a record of your findings.</h2><p>Export the annotated image, counts, measurements and analysis details in PDF or CSV format.</p><div className="export-buttons"><button className="secondary" onClick={report} disabled={reportBusy}><Download size={17}/>{reportBusy?'Preparing…':'Download PDF report'}</button><button className="secondary" onClick={exportCsv} disabled={csvBusy}><FileText size={17}/>{csvBusy?'Exporting…':'Download CSV data'}</button></div></section>}
 </div>

 {/* Export section when pie chart is shown */}
 {chart.length>1&&<div className="detail-grid"><section className="panel report-card"><div className="report-icon"><Download/></div><h2>Keep a record of your findings.</h2><p>Export the annotated image, counts, measurements and analysis details in PDF or CSV format.</p><div className="export-buttons"><button className="secondary" onClick={report} disabled={reportBusy}><Download size={17}/>{reportBusy?'Preparing…':'Download PDF report'}</button><button className="secondary" onClick={exportCsv} disabled={csvBusy}><FileText size={17}/>{csvBusy?'Exporting…':'Download CSV data'}</button></div></section>

 {/* Confidence distribution */}
 {result.detections.length>0&&<section className="panel"><h2>Confidence distribution</h2><p className="muted">Histogram of detection confidences</p><ResponsiveContainer width="100%" height={190}><BarChart data={(() => {
   const bins=[{name:'0-20%',count:0},{name:'20-40%',count:0},{name:'40-60%',count:0},{name:'60-80%',count:0},{name:'80-100%',count:0}];
   result.detections.forEach(d=>{const idx=Math.min(Math.floor(d.confidence*5),4);bins[idx].count++});
   return bins;
 })()} margin={{left:0,right:24,bottom:0}}><CartesianGrid strokeDasharray="4 4" vertical={false}/><XAxis dataKey="name" tick={{fontSize:11}}/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="count" fill="#5b9ea8" radius={[6,6,0,0]} maxBarSize={70}/></BarChart></ResponsiveContainer></section>}
 </div>}

 <section className="panel table-panel"><div className="table-heading"><div><h2>Particle measurements</h2><p className="muted">{result.calibration_um_per_pixel?'Calibrated bounding-box dimensions':'Bounding-box dimensions in pixels'}</p></div><span className="pill">{result.detections.length} detections</span></div><div className="table-scroll"><table><thead><tr><th>PARTICLE</th><th>MORPHOLOGY CLASS</th><th>CONFIDENCE</th><th>WIDTH (PX)</th><th>HEIGHT (PX)</th><th>AREA (PX²)</th>{result.calibration_um_per_pixel&&<th>SIZE (µm)</th>}</tr></thead><tbody>{result.detections.map(d=><tr key={d.particle_id}><td>#{String(d.particle_id).padStart(2,'0')}</td><td><span className="class-indicator" style={{background:CLASS_COLORS[d.class_name]||'#888'}}/>{d.class_name.replaceAll('_',' ')}</td><td><span className="confidence">{pct(d.confidence)}</span></td><td>{d.width_pixels.toFixed(1)}</td><td>{d.height_pixels.toFixed(1)}</td><td>{d.area_pixels.toFixed(1)}</td>{result.calibration_um_per_pixel&&<td>{d.width_um.toFixed(1)} × {d.height_um.toFixed(1)}</td>}</tr>)}</tbody></table>{!result.detections.length&&<p className="muted">No detections at this threshold.</p>}</div></section>

 {/* Scientific disclaimer */}
 <section className="panel disclaimer-panel"><Info size={18}/><div><b>Scientific limitations</b><ul>
  <li>Morphology detection is not polymer identification.</li>
  <li>Visible-light images may contain sediment, bubbles, and natural fibers.</li>
  <li>Performance may vary between microscopes, lighting, and sample matrices.</li>
  <li>Pixel measurements are not physical measurements without calibration.</li>
  <li>Particle count per image is not concentration per litre.</li>
 </ul></div></section>
 </>}
 </>}

 {/* ===== HISTORY TAB ===== */}
 {tab==='history'&&<section className="panel table-panel"><h2>Saved analyses <span className="muted">({history.length})</span></h2><div className="table-scroll"><table><thead><tr><th>IMAGE</th><th>ANALYZED</th><th>MODEL</th><th>PARTICLES</th><th>MEAN CONFIDENCE</th><th>COUNT BAND</th><th/></tr></thead><tbody>{history.map(h=><tr key={h.id}><td>{h.input_name}</td><td>{new Date(h.created_at).toLocaleString()}</td><td>{h.model_name||'Baseline'}</td><td>{h.total_particles}</td><td>{pct(h.average_confidence)}</td><td>{h.contamination_level}</td><td><button className="text-button" onClick={()=>{setResult(h);setFile(null);setCalibration(h.calibration_um_per_pixel??'');setSelectedModel(h.model_id||'baseline');setTab('analyze')}}>View analysis <ArrowUpRight size={14}/></button></td></tr>)}</tbody></table>{!history.length&&<p className="muted">No saved analyses yet. Run your first sample to begin.</p>}</div></section>}

 {/* ===== MODEL TAB ===== */}
 {tab==='model'&&<>{/* Model selector for evaluation */}
 <div className="eval-model-selector"><label>Evaluate model:
  <select value={evalModelId} onChange={e=>{setEvalModelId(e.target.value)}}>
   {models.map(m=><option key={m.model_id} value={m.model_id}>{m.name}{!m.available?' (not available)':''}</option>)}
  </select>
 </label></div>
 <EvaluationPanel modelId={evalModelId}/>
 </>}

 {/* ===== DATASET TAB ===== */}
 {tab==='dataset'&&<DatasetPanel summary={datasetSummary}/>}

 <footer><span><FlaskConical size={14}/> Research prototype · Multi-class microscopic image analysis</span><span>Count bands indicate image detections, not laboratory concentration.</span></footer></div></main></div>
}

/* ===== Evaluation Panel Component ===== */
function EvaluationPanel({modelId}){
  const [evalInfo,setEvalInfo]=useState(null);
  const [loading,setLoading]=useState(true);
  useEffect(()=>{
    setLoading(true);
    api('/evaluation/'+encodeURIComponent(modelId)).then(r=>r.json()).then(d=>{setEvalInfo(d);setLoading(false)}).catch(()=>{setEvalInfo(null);setLoading(false)});
  },[modelId]);

  if(loading) return <section className="panel"><p>Loading evaluation…</p></section>;
  if(!evalInfo) return <section className="panel"><p className="muted">Model not available or metrics not found. The model may still be training.</p></section>;

  const hasMetrics=evalInfo.metrics&&Object.keys(evalInfo.metrics).length>0;

  return <>
  <section className="panel model-intro"><Microscope size={34}/><h2>{evalInfo.model}</h2>
  <p>Trained on microscope images from the <b>{evalInfo.dataset}</b> dataset. Supported classes: <b>{Object.values(evalInfo.classes).join(', ')}</b>.</p>
  <p>Results below measure performance on the supplied validation and held-out test splits. They do not establish performance on other microscopes, sample preparations, or geographic regions.</p>
  {evalInfo.model_id==='baseline'&&<a href="https://huggingface.co/datasets/andreusv/Microplastic-Fragments-WWTP-Dataset" target="_blank" rel="noreferrer">View dataset & attribution <ArrowUpRight size={15}/></a>}
  </section>

  {hasMetrics?Object.entries(evalInfo.metrics).map(([split,m])=><section className="panel metrics-panel" key={split}><h2>{split==='test'?'Held-out test evaluation':split==='val'||split==='validation'?'Validation evaluation':split+' evaluation'}</h2><div className="stats">{[['Precision','metrics/precision(B)'],['Recall','metrics/recall(B)'],['mAP @ 50','metrics/mAP50(B)'],['mAP @ 50–95','metrics/mAP50-95(B)']].map(([label,key])=><div className="stat" key={key}><span>{label}</span><strong>{pct(m[key]??0)}</strong></div>)}</div>

  {/* Per-class metrics if available */}
  {m.per_class&&<div className="per-class-metrics"><h3>Per-class metrics</h3><div className="table-scroll"><table><thead><tr><th>CLASS</th><th>PRECISION</th><th>RECALL</th><th>mAP@50</th><th>mAP@50-95</th></tr></thead><tbody>{Object.entries(m.per_class).map(([cls,cm])=><tr key={cls}><td><span className="class-indicator" style={{background:CLASS_COLORS[cls]||'#888'}}/>{cls}</td><td>{pct(cm.precision??0)}</td><td>{pct(cm.recall??0)}</td><td>{pct(cm['mAP50']??0)}</td><td>{pct(cm['mAP50-95']??0)}</td></tr>)}</tbody></table></div></div>}
  </section>)

  :<section className="panel metrics-panel"><h2>Evaluation metrics</h2><p className="muted">Not evaluated yet. Metrics will appear here after the model has been evaluated on the validation and test sets.</p></section>}

  {/* Model card info */}
  {evalInfo.model_card&&Object.keys(evalInfo.model_card).length>0&&<section className="panel"><h2>Model card</h2><div className="model-card-grid">
   {evalInfo.model_card.architecture&&<div><span className="muted">Architecture</span><strong>{evalInfo.model_card.architecture}</strong></div>}
   {evalInfo.model_card.epochs&&<div><span className="muted">Epochs trained</span><strong>{evalInfo.model_card.epochs}</strong></div>}
   {evalInfo.model_card.batch_size&&<div><span className="muted">Batch size</span><strong>{evalInfo.model_card.batch_size}</strong></div>}
   {evalInfo.model_card.imgsz&&<div><span className="muted">Image size</span><strong>{evalInfo.model_card.imgsz}</strong></div>}
   {evalInfo.model_card.device&&<div><span className="muted">Device</span><strong>{evalInfo.model_card.device}</strong></div>}
   {evalInfo.model_card.training_time&&<div><span className="muted">Training time</span><strong>{evalInfo.model_card.training_time}</strong></div>}
  </div></section>}

  <section className="panel disclaimer-panel"><Info size={18}/><div><p>{evalInfo.disclaimer}</p></div></section>
  </>
}

/* ===== Dataset Panel Component ===== */
function DatasetPanel({summary}){
  if(!summary) return <section className="panel"><p>Loading dataset information…</p></section>;

  const classData=Object.entries(summary.class_distribution||{}).map(([name,count])=>({name:name.replaceAll('_',' '),count}));
  const splitData=Object.entries(summary.splits||{}).map(([name,data])=>({name,images:data.images||0,annotations:data.annotations||0}));
  const totalImages=(summary.total_original_images||0)+(summary.total_synthetic_images||0);

  return <>
  {/* Quick stats */}
  <div className="stats">
   <div className="stat"><div><span>Total images</span><Database size={18}/></div><strong>{totalImages.toLocaleString()}</strong><small>Original: {(summary.total_original_images||0).toLocaleString()} · Synthetic: {(summary.total_synthetic_images||0).toLocaleString()}</small></div>
   <div className="stat"><div><span>Annotated objects</span><Target size={18}/></div><strong>{(summary.total_annotated_objects||0).toLocaleString()}</strong><small>Bounding box annotations</small></div>
   <div className="stat"><div><span>Data sources</span><Layers size={18}/></div><strong>{summary.num_sources||0}</strong><small>{summary.licence_status||'License info unavailable'}</small></div>
   <div className="stat"><div><span>Morphology classes</span><BarChart3 size={18}/></div><strong>{summary.num_classes||0}</strong><small>{(summary.class_names||[]).join(', ')}</small></div>
  </div>

  {/* Dataset splits */}
  {splitData.length>0&&<section className="panel"><h2>Dataset splits</h2><p className="muted">Train / validation / test distribution</p><div className="table-scroll"><table><thead><tr><th>SPLIT</th><th>IMAGES</th><th>ANNOTATIONS</th></tr></thead><tbody>{splitData.map(s=><tr key={s.name}><td style={{textTransform:'capitalize'}}>{s.name}</td><td>{s.images.toLocaleString()}</td><td>{s.annotations.toLocaleString()}</td></tr>)}</tbody></table></div></section>}

  {/* Class distribution chart */}
  {classData.length>0&&<div className="detail-grid"><section className="panel"><h2>Class distribution</h2><p className="muted">Number of annotated objects per class</p><ResponsiveContainer width="100%" height={220}><BarChart data={classData} margin={{left:0,right:24,bottom:0}}><CartesianGrid strokeDasharray="4 4" vertical={false}/><XAxis dataKey="name" tick={{fontSize:12}}/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="count" radius={[6,6,0,0]} maxBarSize={80}>{classData.map((entry,i)=><Cell key={i} fill={CLASS_COLORS[entry.name.replaceAll(' ','_')]||PIE_COLORS[i%PIE_COLORS.length]}/>)}</Bar></BarChart></ResponsiveContainer></section>

  <section className="panel"><h2>Class proportions</h2><p className="muted">Relative share of each morphology class</p><ResponsiveContainer width="100%" height={220}><PieChart><Pie data={classData} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name,percent})=>`${name} ${(percent*100).toFixed(0)}%`}>{classData.map((entry,i)=><Cell key={i} fill={CLASS_COLORS[entry.name.replaceAll(' ','_')]||PIE_COLORS[i%PIE_COLORS.length]}/>)}</Pie><Tooltip/></PieChart></ResponsiveContainer></section></div>}

  {/* Data sources table */}
  {(summary.sources||[]).length>0&&<section className="panel table-panel"><h2>Data sources</h2><div className="table-scroll"><table><thead><tr><th>SOURCE</th><th>LICENSE</th><th>IMAGES</th><th>ANNOTATIONS</th><th>TYPE</th></tr></thead><tbody>{summary.sources.map((s,i)=><tr key={i}><td>{s.name}</td><td>{s.license}</td><td>{(s.images||0).toLocaleString()}</td><td>{(s.annotations||0).toLocaleString()}</td><td>{(s.type||'').replaceAll('_',' ')}</td></tr>)}</tbody></table></div></section>}

  <section className="panel disclaimer-panel"><Info size={18}/><div><b>Dataset notes</b><ul>
   <li>DeepParticle images are cropped individual particles with approximate bounding boxes (~82% centered).</li>
   <li>Synthetic images are generated for training augmentation and are not real microscopy samples.</li>
   <li>Original image counts exclude augmented copies — augmented images are never counted as original.</li>
   <li>Some classes have fewer samples, which may affect model performance for those classes.</li>
  </ul></div></section>
  </>
}
