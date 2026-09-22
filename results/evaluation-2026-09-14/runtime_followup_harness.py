from pathlib import Path
exec(Path(__file__).with_name('runtime_harness.py').read_text().split('\nresults=[]')[0])
results=[]
for args in [('hd_cnn_repeat',TMP/'hd.mp4',8203,['--mode','cnn','--prebuffer-mb','0']),('short_low_buffer',TMP/'tiny.mp4',8204,['--mode','cnn','--prebuffer-mb','0'])]:
 r=case(*args); results.append(r);print(json.dumps(r),flush=True)
 (OUT/'runtime_followup.json').write_text(json.dumps(results,indent=2))
