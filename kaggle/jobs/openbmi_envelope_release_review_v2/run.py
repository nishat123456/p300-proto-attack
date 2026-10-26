"""Cloud-only pre-aggregation review for disclosed post-execution numerical envelope."""
import os,sys,json,zlib,base64,hashlib,subprocess,math,csv,time,unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def find(name):
 p=list(Path('/kaggle/input').rglob(name));assert len(p)==1;return p[0]
def read(name):return json.loads(zlib.decompress(find(name).read_bytes()))
manifest=read('review_manifest.zlib');payload=read('source_payload.zlib');code=Path('/kaggle/working/code');code.mkdir(exist_ok=True)
for name,v in payload.items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v));assert digest(p)==manifest['code_sha256'][name]
os.environ['PYTHONPATH']=str(code/'src')+':'+str(code);sys.path[:0]=[str(code/'src'),str(code)]
from p3a.data.openbmi_release import execution_code_hashes
from reporting.numerical_envelope_contract import check_budget,CONTRACT
from p3a.defenses.clean_calibration import METHODS
from p3a.eval.reliability_metrics import clean_rejection_accounting
import numpy as np
protocol=code/'splits/openbmi_independent_20261022/protocol.json';ph=digest(protocol);assert ph==manifest['protocol_sha256']
receipt=json.loads((protocol.parent/'receipt_s37.json').read_text());assert execution_code_hashes(code)==receipt['execution_code_sha256']
class Bounds(unittest.TestCase):
 def test_strict(self):self.assertFalse(check_budget(.5,.5))
 def test_upward(self):self.assertTrue(check_budget(float(np.float32(.2)),.2))
 def test_above(self):
  with self.assertRaises(ValueError):check_budget(.200001,.2)
 def test_negative(self):
  with self.assertRaises(ValueError):check_budget(-1,.2)
 def test_nonfinite(self):
  with self.assertRaises(ValueError):check_budget(float('nan'),.2)
 def test_downward(self):
  e=.3;self.assertFalse(check_budget(e,e))
res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Bounds));assert res.wasSuccessful()
# Original portable suite, original frozen inventory, unchanged four historical exclusions.
args=[sys.executable,'-m','pytest','-q','--disable-warnings','tests']+[f'--ignore=tests/{n}' for n in ['test_comp3.py','test_letter_head.py','test_bank_report.py','test_kaggle_kill_test_run.py']]
proc=subprocess.run(args,cwd=code,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(out/'portable_original_inventory.txt').write_text(proc.stdout);assert proc.returncode==0,'original portable regression failed'
entries=read('private_transfer.zlib');assert len(entries)==3978
root=Path('/kaggle/working/archive');root.mkdir(exist_ok=True)
def fetch(item):
 name=item['name'];rel=Path(name)
 assert not rel.is_absolute() and '..' not in rel.parts and name.startswith('independent_results/task')
 assert name.endswith('.json') or ('validation_clean_' in name and name.endswith('.csv'))
 p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   r=requests.get(item['url'],timeout=(30,180));r.raise_for_status();p.write_bytes(r.content);return
  except Exception as e:
   if attempt==3:raise RuntimeError('primary archive transport failure '+name+' '+type(e).__name__) from None
   time.sleep(2**attempt)
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,entries))
del entries
for rel,h in manifest['expected_compact_sha256'].items():assert digest(root/rel)==h,'original compact hash mismatch '+rel
rows=0;strict=0;maxrel=0.;maxabs=0.;per=[];csv_hashes={}
for task in manifest['tasks']:
 t=task['task'];s=task['subject'];seed=task['seed'];d=root/f'independent_results/task{t:02d}/out'
 complete=json.loads((d/'task_complete.json').read_text());assert complete['task']==t and complete['subject']==s and complete['seed']==seed
 lp=list(d.rglob('subject_execution.json'));assert len(lp)==1;l=json.loads(lp[0].read_text());assert len(l['completed'])==38 and l['binding']['protocol_sha256']==ph
 for stage,outputs in l['completed'].items():
  for name,h in outputs.items():
   if Path(name).suffix=='.pt':continue
   assert digest(lp[0].parent/name)==h
 cp=list(d.rglob(f'validation_clean_s{s:02d}_seed{seed}.json'));assert len(cp)==1;cp=cp[0];clean=json.loads(cp.read_text())
 assert clean['protocol_sha256']==ph
 cal=list(d.rglob(f'calibration_s{s:02d}_seed{seed}.json'));assert len(cal)==1;cal=json.loads(cal[0].read_text())
 with cp.with_suffix('.csv').open() as f:cr=list(csv.DictReader(f))
 assert len(cr)==36*7*3
 for summary in clean['summaries']:
  m=summary['method'];alpha=summary['nominal_far'];rr=[r for r in cr if r['method']==m and float(r['nominal_far'])==alpha]
  assert len(rr)==36 and len({r['trial_id'] for r in rr})==36
  for r in rr:
   assert int(r['subject'])==s and int(r['seed'])==seed
   score=float(r['anomaly']);threshold=float(r['threshold']);assert math.isfinite(score) and math.isfinite(threshold)
   assert threshold==cal['methods'][m]['thresholds'][str(alpha)]
   assert r['rejected'] in ['True','False'] and (r['rejected']=='True')==(score>threshold)
   assert r['correct'] in ['True','False'] and (r['correct']=='True')==(r['predicted_symbol']==r['intended_symbol'])
  counted=clean_rejection_accounting(np.array([r['correct']=='True' for r in rr]),np.array([r['rejected']=='True' for r in rr]))
  for k,v in counted.items():
   if v is None:assert summary[k] is None
   else:assert math.isclose(v,summary[k],abs_tol=1e-12),k
 csv_hashes[str(cp.with_suffix('.csv').relative_to(root))]=digest(cp.with_suffix('.csv'))
 ids=set();count=0
 for p in d.rglob(f'attack_audit_seed{seed}.json'):
  a=json.loads(p.read_text());identity=a['trial_id'];assert identity not in ids;ids.add(identity)
  assert a['subject']==s and a['seed']==seed and a['protocol_sha256']==ph and a['manifest_sha256']==clean['manifest_sha256']
  assert len(a['rows'])==(34 if identity.endswith(':000') else 27)
  for row in a['rows']:
   v,e=row['applied_linf'],row['raw_epsilon'];bad=check_budget(v,e);strict+=int(bad);rows+=1;count+=1
   if bad:maxrel=max(maxrel,(v-e)/e);maxabs=max(maxabs,v-e)
 assert ids=={f'openbmi_candidate:s{s:02d}_session1:test:{i:03d}' for i in range(36)}
 per.append(dict(**task,setting_rows=count,ledger_sha256=digest(lp[0])))
assert rows==49929 and strict==13419
proof=dict(status='review_passed_no_combined_performance_computed',contract=CONTRACT,protocol_sha256=ph,original_execution_revision=manifest['original_execution_revision'],tasks=per,rows=rows,strict_violations_retained=strict,max_relative_excess=maxrel,max_absolute_excess=maxabs,guard_tests=res.testsRun,original_portable_suite_returncode=proc.returncode,source_sha256=manifest['code_sha256'],original_json_sha256=manifest['expected_compact_sha256'],clean_csv_sha256=csv_hashes,runner_sha256=digest(__file__),review_manifest_sha256=digest(find('review_manifest.zlib')),limitations='Post-execution revised numerical contract; original strict failure retained; paired stability only two tasks; no equivalence claim. Receipt inspection and private preservation required before aggregate execution.')
(out/'analysis_release_receipt.json').write_text(json.dumps(proof,indent=2))
(out/'review_complete.md').write_text('# Numerical envelope release review\n\nAll51originaltasks and51cleanCSV verifiedcloud-to-cloud against savedcompactJSON andstagehashes; cleanCSVthreshold/decision/summaryconsistency checked. Exact envelope passes49929rows while13419strictviolations remain disclosed. No combinedperformancecomputed. Reviewreceiptmustbeinspectedandpreservedbeforeanalysis.\n')
print('New analysis contract review passed; receipt written. No combined performance aggregation.',flush=True)
