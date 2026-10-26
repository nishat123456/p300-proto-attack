import json,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict
import torch
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
paths=list(Path('/kaggle/input').rglob('strict_budget_diagnostic.json'));assert len(paths)==1
p=paths[0];d=json.loads(p.read_text());v=d['violations']
summary=dict(shards=d['shards'],rows=d['rows'],strict_violations=d['strict_violations'],nonfinite=d['nonfinite_rows'],all_equal_float32_epsilon=all(r['applied_equals_float32_epsilon'] for r in v),max_relative_excess=max(r['relative_excess'] for r in v),max_absolute_excess=max(r['excess'] for r in v),affected_subjects=sorted({r['subject'] for r in v}),affected_subject_seed_tasks=sorted({(r['subject'],r['seed']) for r in v}),by_attack_budget=d['by_attack_budget'],source_report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='numerical validity only, not performance or harmlessness determination')
def strict_project(eeg,delta,epsilon):
 if eeg.shape!=delta.shape or not math.isfinite(epsilon) or epsilon<0:raise ValueError('aligned finite nonnegative budget required')
 with torch.no_grad():
  limit=torch.as_tensor(epsilon,dtype=eeg.dtype,device=eeg.device)
  if float(limit)>epsilon:limit=torch.nextafter(limit,torch.zeros_like(limit))
  applied=eeg+delta.clamp(-limit,limit)
  for _ in range(4):
   excess=(applied-eeg).abs().double()>epsilon
   if not excess.any():break
   applied=torch.where(excess,torch.nextafter(applied,eeg),applied)
  result=applied-eeg
  if not torch.isfinite(result).all() or bool((result.abs().double()>epsilon).any()):raise ValueError('strict float64 budget failure')
  return result
checks=[]
# Test every offending declared epsilon against varied raw offsets and both signs.
for epsilon in sorted({r['raw_epsilon'] for r in v}):
 raw=torch.tensor([0.,1.,-1.,100.,-100.,10000.,-10000.],dtype=torch.float32)
 for sign in [-1,1]:
  actual=(raw+strict_project(raw,torch.full_like(raw,sign*epsilon*2),epsilon))-raw
  assert bool((actual.double().abs()<=epsilon).all())
 checks.append(epsilon)
# Deterministic synthetic scale/offset sweep independent of target outcomes.
g=torch.Generator().manual_seed(20261013);n=0
for baseline in [0.,1.,100.,10000.,1e8]:
 for epsilon in [0.,1e-8,.001,.05,.1,.2,.3,.5,4.5227703094482425,100.]:
  raw=torch.randn(4,31,generator=g)*baseline;delta=torch.randn(4,31,generator=g)*(epsilon*4)
  result=strict_project(raw,delta,epsilon);assert bool((((raw+result)-raw).double().abs()<=epsilon).all());n+=1
summary['candidate_projection_checks']=dict(unique_offending_epsilons=len(checks),synthetic_scale_offset_cases=n,status='pass',candidate_only=True,scientific_source_inventory_modified=False)
(out/'budget_summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Strict budget diagnostic summary','',f"Inspected{d['shards']} character shards and{d['rows']} setting rows. Strict violations:{len(v)}; nonfinite:{d['nonfinite_rows']}.",f"All violations equal float32 rounding of declared epsilon:{summary['all_equal_float32_epsilon']}.",f"Maximum relative excess:{summary['max_relative_excess']:.12g}; maximum absolute excess:{summary['max_absolute_excess']:.12g}.",'','This does not establish harmlessness. Original strict protocol rejects these settings. No performance outcomes were analyzed or bounds relaxed.','',f'Candidate inward float32 projection with float64 verification passed{len(checks)} offending epsilon cases and{n} synthetic scale/offset cases. Candidate has not been applied to experiments.','', 'Any correction requires prospective provenance, preservation of originals and valid rerun accounting.']
(out/'budget_summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
