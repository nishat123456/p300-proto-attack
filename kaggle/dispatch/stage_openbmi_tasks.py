from pathlib import Path
import argparse,json,base64,zlib,subprocess
r=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Package fixed task indices only; no experiment or launch')
parser.add_argument('--task-index',type=int,nargs='+',required=True)
args=parser.parse_args()
if len(set(args.task_index))!=len(args.task_index) or any(not 0<=i<51 for i in args.task_index):raise ValueError('distinct frozen task indices0..50 required')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
dispatch=json.loads((r/'configs/openbmi_five_account_dispatch_candidate.json').read_text())
for task in [dispatch['tasks'][i] for i in sorted(args.task_index)]:
 index=task['task_index'];subject=task['subject'];seed=task['seed'];owner=task['account']
 folder=r/f'kaggle/jobs/openbmi_independent_task{index:02d}';folder.mkdir(exist_ok=False)
 files={str(p.relative_to(r)):p.read_text() for sub in ('src','scripts') for p in (r/sub).rglob('*.py') if '__pycache__' not in p.parts}
 for name in ('protocol.json',f'manifest_s{subject:02d}.json',f'receipt_s{subject:02d}.json'):
  path=r/'splits/openbmi_independent_20261022'/name;files[str(path.relative_to(r))]=path.read_text()
 payload=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
 runner="""import os,sys,json,zlib,base64,subprocess,hashlib,requests
from pathlib import Path
import torch
assert torch.cuda.is_available(),'GPU required; do not silently execute benchmark on CPU'
code=Path('/kaggle/working/code');code.mkdir()
for name,value in json.loads(zlib.decompress(base64.b64decode(PAYLOAD))).items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
os.environ['PYTHONPATH']=str(code/'src')+':'+str(code)
sys.path[:0]=[str(code/'src'),str(code)]
out=Path('/kaggle/working/out');out.mkdir()
subject=SUBJECT;seed=SEED
root=Path('/kaggle/input')
def unique(name):
 found=list(root.rglob(name))
 assert len(found)==1,(name,len(found))
 return found[0]
raw=Path('/kaggle/working')/f'sess01_subj{subject:02d}_EEG_ERP.mat'
url=f'https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/session1/s{subject}/'+raw.name
with requests.get(url,stream=True,timeout=(30,300)) as response:
 response.raise_for_status()
 with raw.open('wb') as handle:
  for chunk in response.iter_content(1024*1024):handle.write(chunk)
release=code/'splits/openbmi_independent_20261022'
from p3a.data.openbmi_release import load_execution_release
raw_sha=hashlib.sha256(raw.read_bytes()).hexdigest()
load_execution_release(release/'protocol.json',release/f'manifest_s{subject:02d}.json',release/f'receipt_s{subject:02d}.json',subject,seed,raw_sha)
(out/'task_start.json').write_text(json.dumps(dict(task=TASK,subject=subject,seed=seed,account=OWNER,git_commit=REVISION,raw_sha256=raw_sha,scope='prespecified independent evaluation; no tuning'),indent=2))
command=[sys.executable,str(code/'scripts/run_openbmi_independent_subject.py'),
 '--protocol',str(release/'protocol.json'),'--manifest',str(release/f'manifest_s{subject:02d}.json'),
 '--release-receipt',str(release/f'receipt_s{subject:02d}.json'),'--subject',str(subject),'--seed',str(seed),
 '--checkpoint',str(unique(f'source_seed{seed}.pt')),'--references',str(unique(f'references_seed{seed}.npz')),
 '--reference-json',str(unique(f'references_seed{seed}.json')),'--template',str(unique(f'source_template_seed{seed}.pt')),
 '--raw-file',str(raw),'--schedule',str(unique('schedule.mat')),'--out-dir',str(out/f's{subject:02d}_seed{seed}'),
 '--git-commit',REVISION]
subprocess.run(command,check=True)
(out/'task_complete.json').write_text(json.dumps(dict(task=TASK,subject=subject,seed=seed,account=OWNER,git_commit=REVISION,scope='all36 original characters, frozen settings'),indent=2))
raw.unlink()
"""
 for key,value in dict(PAYLOAD=repr(payload),SUBJECT=str(subject),SEED=str(seed),TASK=str(index),OWNER=repr(owner),REVISION=repr(revision)).items():runner=runner.replace(key,value)
 (folder/'run.py').write_text(runner)
 (folder/'kernel-metadata.json').write_text(json.dumps(dict(id=f'{owner}/p3a-independent-t{index:02d}',title=f'p3a-independent-t{index:02d}',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],kernel_sources=['mustaqimnishat/p3a-openbmi-source','mustaqimnishat/p3a-openbmi-refcal']),indent=2)+'\n')
 (folder/'payload-manifest.json').write_text(json.dumps(dict(task=task,source_commit=revision,files=sorted(files),release='splits/openbmi_independent_20261022',scope='independent frozen execution; same code inventory as cloud release'),indent=2)+'\n')
