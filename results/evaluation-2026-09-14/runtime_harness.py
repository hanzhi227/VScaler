import subprocess as sp, pathlib, time, json, signal, urllib.request, socket
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'results/evaluation-2026-09-14'
TMP=pathlib.Path('/tmp/video-upscaler-audit'); TMP.mkdir(exist_ok=True)
FF='/opt/homebrew/bin/ffmpeg'; FP='/opt/homebrew/bin/ffprobe'
ENG=str(ROOT/'engine/AnimeUpscaler/.build/release/animeupscale')
def run(args):
 r=sp.run(args,capture_output=True,text=True,timeout=60)
 if r.returncode: raise RuntimeError(r.stderr[-2000:])
 return r.stdout
run([FF,'-v','error','-f','lavfi','-i','testsrc2=size=320x180:rate=24','-t','1','-c:v','libx264','-y',str(TMP/'tiny.mp4')])
run([FF,'-v','error','-f','lavfi','-i','testsrc2=size=720x480:rate=24','-t','8','-vf','setsar=32/27','-c:v','libx264','-y',str(TMP/'anamorphic.mp4')])
run([FF,'-v','error','-i',str(ROOT/'test-media/segments/synth_clean_1080p.mp4'),'-t','8','-c','copy','-y',str(TMP/'hd.mp4')])
def status(port):
 with urllib.request.urlopen(f'http://127.0.0.1:{port}/status.json',timeout=1) as r: return json.load(r)
def probe(p):
 return json.loads(run([FP,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,sample_aspect_ratio,display_aspect_ratio,avg_frame_rate,nb_read_frames,color_space,color_transfer,color_primaries','-count_frames','-of','json',str(p)]))['streams']
def case(name,source,port,extra,read=True):
 with open(OUT/(name+'.jsonl'),'w') as log,open(OUT/(name+'.stderr'),'w') as err:
  p=sp.Popen([ENG,'--file',str(source),'--port',str(port),*extra],stdout=log,stderr=err)
  result={'source':str(source),'options':extra}
  try:
   deadline=time.monotonic()+10
   while True:
    if p.poll() is not None: raise RuntimeError(f'engine exited {p.returncode}')
    try:
     s=status(port)
     if s.get('sessionId',0)>0: break
    except (OSError,ValueError): pass
    if time.monotonic()>deadline: raise TimeoutError('startup')
    time.sleep(.05)
   t0=time.monotonic()
   if read:
    captured=TMP/(name+'.ts'); n=0
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/stream.ts?session=1',timeout=5) as r,open(captured,'wb') as f:
      while True:
       b=r.read(65536)
       if not b: break
       n+=len(b); f.write(b)
     result['capture']='eof'
    except (OSError,TimeoutError) as e: result['capture']=str(e)
    result['bytes']=n; result['capture_seconds']=time.monotonic()-t0
    if n: result['output_probe']=probe(captured)
   else: time.sleep(7)
   result['status']=status(port)
   result['source_probe']=probe(source)
  except Exception as e: result['error']=repr(e)
  finally:
   if p.poll() is None:
    p.send_signal(signal.SIGINT)
    try:p.wait(timeout=10)
    except sp.TimeoutExpired:p.kill();p.wait()
   result['exitcode']=p.returncode
 return result
results=[]
for args in [('short_default',TMP/'tiny.mp4',8193,['--mode','cnn']),('anamorphic',TMP/'anamorphic.mp4',8194,['--mode','bypass','--prebuffer-mb','0']),('hd_cnn',TMP/'hd.mp4',8195,['--mode','cnn','--prebuffer-mb','0']),('subtitle_toggle',ROOT/'test-media/segments/subs_text.mkv',8196,['--mode','bypass','--script','1:subs-on:0,5:subs-off'],False)]:
 r=case(*args);results.append(r);print(json.dumps(r),flush=True)
 (OUT/'runtime.json').write_text(json.dumps(results,indent=2))
