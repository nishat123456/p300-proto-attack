import base64,zlib,json,copy,importlib.util,subprocess,sys,hashlib
from pathlib import Path
reporter=Path('/kaggle/working/build_cost_report.py')
reporter.write_bytes(zlib.decompress(base64.b64decode('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')))
root=Path('/kaggle/input');out=Path('/kaggle/working/out');out.mkdir()
clean_paths=[p for seed in [20261013,20261014,20261015] for p in root.rglob(f'validation_clean_s19_seed{seed}.json')]
time_paths=[p for seed in [20261013,20261014,20261015] for p in root.rglob(f'source_cost_seed{seed}.json')]
clean=[json.loads(p.read_text()) for p in clean_paths];timing=[json.loads(p.read_text()) for p in time_paths]
ph=clean[0]['protocol_sha256'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocols=[p for p in root.rglob('protocol.json') if sha(p)==ph];assert len(protocols)==1
spec=importlib.util.spec_from_file_location('costreport',reporter);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
assert mod.validate(clean,timing,ph,True)==[19]
checks=['valid complete development cost inputs accepted']
for label,c,t,h,dev in [
 ('single user refused for independent',clean,timing,ph,False),
 ('missing clean seed refused',clean[:-1],timing,ph,True),
 ('missing timing seed refused',clean,timing[:-1],ph,True),
 ('wrong protocol refused',clean,timing,'0'*64,True)]:
 try:mod.validate(c,t,h,dev)
 except ValueError:checks.append(label)
 else:raise AssertionError(label)
for label,mutate in [
 ('reference mismatch refused',lambda c:c[0].update(reference_sha256='0'*64)),
 ('missing marker refused',lambda c:c[0]['marker_duration_seconds'].pop()),
 ('retry inconsistency refused',lambda c:c[0]['summaries'][0].update(iid_retry_overhead=123)),
 ('nominal omission refused',lambda c:c[0]['summaries'].pop())]:
 c=copy.deepcopy(clean);mutate(c)
 try:mod.validate(c,timing,ph,True)
 except ValueError:checks.append(label)
 else:raise AssertionError(label)
subprocess.run([sys.executable,str(reporter),'--clean-root',str(root),'--timing-root',str(root),'--protocol',str(protocols[0]),'--out-dir',str(out/'development_cost'),'--git-commit','b2cb47532aa6079d41ceb359a308e4abdf63b4bd','--development-only'],check=True)
(out/'cost_report_checks.json').write_text(json.dumps(dict(checks=checks,git_commit='b2cb47532aa6079d41ceb359a308e4abdf63b4bd',reporter_sha256=sha(reporter),scope='nine cloud checks and real participant19 descriptive cost report; not independent population results'),indent=2))
print('completed',len(checks),'cost report checks',flush=True)
