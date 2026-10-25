import os,sys,json,base64,zlib,hashlib,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
CODE_BLOB='__CODE_BLOB__'
TRANSFER_BLOB='__TRANSFER_BLOB__'
revision='0959347d074e2a7ab5720c39946d8f28bfd875d9'
code=Path('/kaggle/working/code');code.mkdir(exist_ok=True)
for name,value in json.loads(zlib.decompress(base64.b64decode(CODE_BLOB))).items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
os.environ['PYTHONPATH']=str(code/'src')+':'+str(code);sys.path[:0]=[str(code/'src'),str(code)]
from p3a.data.openbmi_release import execution_code_hashes
protocol=code/'splits/openbmi_independent_20261022/protocol.json'
receipt=json.loads((protocol.parent/'receipt_s37.json').read_text())
assert execution_code_hashes(code)==receipt['execution_code_sha256'],'frozen source inventory mismatch'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert digest(protocol)=='a5699d95b0d35343da207dec8d657f380d1fd49a7664424682cd6e7c910b9916'
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
(out/'runner_provenance.json').write_text(json.dumps(dict(source_commit=revision,runner_sha256=digest(__file__),protocol_sha256=digest(protocol),seed=20261013,scope='complete independent51 task cloud aggregation; fixed existing scientific inventory'),indent=2))
records=json.loads(zlib.decompress(base64.b64decode(TRANSFER_BLOB)))
assert sorted(r['task'] for r in records)==list(range(51)),'all51 unique task inventories required'
root=Path('/kaggle/working/independent_results');root.mkdir(exist_ok=True)
def fetch(item):
 task,name,url=item
 path=Path(name)
 assert not path.is_absolute() and '..' not in path.parts
 assert name.startswith('out/') and (name.endswith('.json') or (path.name.startswith('validation_clean_') and name.endswith('.csv')))
 target=root/f'task{task:02d}'/path;target.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   r=requests.get(url,timeout=(30,180));r.raise_for_status();target.write_bytes(r.content);return
  except Exception:
   if attempt==3:raise RuntimeError(f'cloud transfer failed task{task:02d} file{name}') from None
   time.sleep(2**attempt)
items=[(r['task'],f['name'],f['url']) for r in records for f in r['files']]
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,items))
# Do not emit private signed URLs in logs or preserved reports.
del records,items
proof=[]
for task in range(51):
 taskroot=root/f'task{task:02d}'/'out';complete=json.loads((taskroot/'task_complete.json').read_text());assert complete['task']==task
 ledgers=list(taskroot.rglob('subject_execution.json'));assert len(ledgers)==1
 ledgerpath=ledgers[0];ledger=json.loads(ledgerpath.read_text());assert ledger['binding']['protocol_sha256']==digest(protocol)
 assert ledger['status']=='all36 original clean/attack characters executed; no outcome-selected omissions'
 assert len(ledger['completed'])==38
 checked=0
 for stage,outputs in ledger['completed'].items():
  for name,sha in outputs.items():
   candidate=ledgerpath.parent/name
   if candidate.suffix=='.pt':continue # User preprocessing stays on original private kernel; no EEG/model transfer.
   assert candidate.is_file(),f'missing transferred stage output {task} {name}'
   assert digest(candidate)==sha,f'stage output hash mismatch {task} {name}'
   checked+=1
 proof.append(dict(task=task,subject=complete['subject'],seed=complete['seed'],ledger_sha256=digest(ledgerpath),checked_compact_files=checked))
(out/'cloud_collection_proof.json').write_text(json.dumps(dict(tasks=proof,scope='51 source ledgers and compact JSON plus clean CSV copied directly cloud-to-cloud; PT/raw excluded'),indent=2))
def run(args):subprocess.run([sys.executable]+args,cwd=code,check=True)
run(['scripts/aggregate_openbmi_reliability.py','--root',str(root),'--protocol',str(protocol),'--out-dir',str(out/'aggregate'),'--git-commit',revision,'--seed','20261013'])
run(['reporting/build_reliability_report.py','--aggregate',str(out/'aggregate/reliability_aggregate.json'),'--protocol',str(protocol),'--out-dir',str(out/'report'),'--git-commit',revision])
run(['reporting/build_cost_report.py','--clean-root',str(root),'--timing-root',str(code/'timing'),'--protocol',str(protocol),'--out-dir',str(out/'cost_report'),'--git-commit',revision])
(out/'aggregation_complete.json').write_text(json.dumps(dict(tasks=51,eligible_subjects=17,source_commit=revision,protocol_sha256=digest(protocol),scope='complete frozen independent aggregation and reporting; visual QA and manuscript interpretation pending'),indent=2))
print('Complete51 task aggregation and reports finished; raw/models never transferred',flush=True)
