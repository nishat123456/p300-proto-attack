"""Kaggle-only descriptive paired sensitivity analysis. Does not amend strict primary aggregation."""
import json,math,struct,hashlib,unittest
from pathlib import Path
from collections import defaultdict,Counter
SEED=20261013
TASKS=(10,15)
METHODS=('confidence','entropy','bu','lid','md','adjust_md','kde')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def budget_state(r):
 e=r['raw_epsilon'];v=r['applied_linf']
 if not math.isfinite(e) or not math.isfinite(v) or e<0 or v<0:raise ValueError('invalid norm')
 upper=max(e,f32(e))
 if v>upper:raise ValueError('outside exact float32 conversion envelope')
 return v>e
PARAMS=('steps','restarts','source_updates','detector','weight')
def key(r):return (r['attack'],r['budget'],json.dumps({k:r['settings'][k] for k in PARAMS if k in r['settings']},sort_keys=True))
def indexed(rows):
 d={}
 for r in rows:
  k=key(r)
  if k in d:raise ValueError('duplicate setting')
  d[k]=r
 return d
class Guards(unittest.TestCase):
 def test_strict(self):self.assertFalse(budget_state(dict(raw_epsilon=.5,applied_linf=.5)))
 def test_rounding(self):self.assertTrue(budget_state(dict(raw_epsilon=.2,applied_linf=f32(.2))))
 def test_reject(self):
  with self.assertRaises(ValueError):budget_state(dict(raw_epsilon=.2,applied_linf=.200001))
 def test_nonfinite(self):
  with self.assertRaises(ValueError):budget_state(dict(raw_epsilon=.2,applied_linf=float('nan')))
 def test_duplicate(self):
  r=dict(attack='FGSM',budget=.2,settings={'steps':1})
  with self.assertRaises(ValueError):indexed([r,r])
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards));assert tests.wasSuccessful()
mps=list(Path('/kaggle/input').rglob('input_manifest.json'));assert len(mps)==1
mp=mps[0];root=mp.parent;manifest=json.loads(mp.read_text())
assert manifest['tasks']==list(TASKS)
for rel,h in manifest['sha256'].items():assert sha(root/rel)==h
allgroups=defaultdict(Counter);detail=[];tot=Counter();scores=defaultdict(float);lineage=[];own=defaultdict(Counter);unions=defaultdict(lambda:{'original':[],'corrective':[]})
for task in TASKS:
 dirs={v:root/f'task{task:02d}_{v}' for v in ('original','corrective')}
 ledgers={v:json.loads(next(dirs[v].rglob('subject_execution.json')).read_text()) for v in dirs}
 for v,l in ledgers.items():assert len(l['completed'])==38
 assert ledgers['corrective']['binding']['git_commit']=='95ca75c0e80709d53ab22e1bf2ecbc3f40dc18f3'
 old={json.loads(p.read_text())['trial_id']:p for p in dirs['original'].rglob('attack_audit_seed*.json')}
 new={json.loads(p.read_text())['trial_id']:p for p in dirs['corrective'].rglob('attack_audit_seed*.json')}
 assert len(old)==36 and set(old)==set(new)
 taskcount=Counter();changed=[]
 for identity in sorted(old):
  a=json.loads(old[identity].read_text());b=json.loads(new[identity].read_text())
  for field in ('subject','seed','trial_id','target','intended','protocol_sha256','manifest_sha256','checkpoint_sha256','reference_sha256','template_sha256','raw_sha256','schedule_sha256','gradient_draws','verification_draws','verification_seed'):
   assert a[field]==b[field],field
  aa=indexed(a['rows']);bb=indexed(b['rows']);assert aa.keys()==bb.keys()
  assert len(aa)==(34 if identity.endswith(':000') else 27)
  for k in sorted(aa):
   x,y=aa[k],bb[k];assert x['raw_epsilon']==y['raw_epsilon']
   violation=budget_state(x);assert not budget_state(y)
   assert set(x['scores'])==set(y['scores'])==set(x['rejected_at_nominal5'])==set(y['rejected_at_nominal5'])
   metrics=dict(rows=1,original_strict_violations=int(violation),predicted_flips=int(x['predicted']!=y['predicted']),targeted_success_flips=int(x['targeted_success']!=y['targeted_success']),original_targeted=int(x['targeted_success']),corrective_targeted=int(y['targeted_success']))
   taskcount.update(metrics);tot.update(metrics)
   if metrics['predicted_flips'] or metrics['targeted_success_flips']:changed.append(dict(trial_id=identity,setting=k,predicted_old=x['predicted'],predicted_new=y['predicted']))
   for m in sorted(x['scores']):
    assert math.isfinite(x['scores'][m]) and math.isfinite(y['scores'][m])
    g=(x['attack'],x['budget'],m);c=allgroups[g];c['paired_rows']+=1
    c['rejection_flips']+=int(x['rejected_at_nominal5'][m]!=y['rejected_at_nominal5'][m])
    c['original_accepted_targeted']+=int(x['targeted_success'] and not x['rejected_at_nominal5'][m])
    c['corrective_accepted_targeted']+=int(y['targeted_success'] and not y['rejected_at_nominal5'][m])
    c['accepted_targeted_flips']+=int((x['targeted_success'] and not x['rejected_at_nominal5'][m])!=(y['targeted_success'] and not y['rejected_at_nominal5'][m]))
    scores[g]=max(scores[g],abs(x['scores'][m]-y['scores'][m]))
    if x['attack']!='detector_aware' or x['settings']['detector']==m:
     z=own[g];z['paired_recipe_rows']+=1
     z['rejection_flips']+=int(x['rejected_at_nominal5'][m]!=y['rejected_at_nominal5'][m])
     z['accepted_targeted_flips']+=int((x['targeted_success'] and not x['rejected_at_nominal5'][m])!=(y['targeted_success'] and not y['rejected_at_nominal5'][m]))
     u=unions[(task,identity,x['attack'],x['budget'],m)]
     u['original'].append(bool(x['targeted_success'] and not x['rejected_at_nominal5'][m]))
     u['corrective'].append(bool(y['targeted_success'] and not y['rejected_at_nominal5'][m]))
 detail.append(dict(task=task,subject=a['subject'],seed=a['seed'],counts=dict(taskcount),changed_character_settings=changed))
 lineage.append(dict(task=task,original_git=ledgers['original']['binding']['git_commit'],corrective_git=ledgers['corrective']['binding']['git_commit']))
groups=[dict(attack=k[0],budget=k[1],method=k[2],**c,max_absolute_score_difference=scores[k]) for k,c in sorted(allgroups.items())]
final=defaultdict(Counter)
for (task,identity,attack,budget,m),u in unions.items():
 expected=2 if attack=='detector_aware' and identity.endswith(':000') else 1
 assert len(u['original'])==len(u['corrective'])==expected
 a,b=any(u['original']),any(u['corrective']);c=final[(attack,budget,m)]
 c['character_pairs']+=1;c['original_accepted_targeted']+=int(a);c['corrective_accepted_targeted']+=int(b);c['accepted_targeted_flips']+=int(a!=b)
assert len(final)==147 and all(c['character_pairs']==72 for c in final.values())
result=dict(own_detector_recipe_comparisons=[dict(attack=k[0],budget=k[1],method=k[2],**c) for k,c in sorted(own.items())],own_detector_character_union=[dict(attack=k[0],budget=k[1],method=k[2],**c) for k,c in sorted(final.items())],status='complete',analysis_seed=SEED,prospective_scope=manifest['analysis_scope'],tasks=list(TASKS),tests_passed=tests.testsRun,input_manifest_sha256=sha(mp),runner_sha256=sha(Path(__file__)),totals=dict(tot),by_task=detail,by_attack_budget_method=groups,lineage=lineage,interpretation='Descriptive paired full-run differences on two availability-selected completed corrective tasks. Not a causal isolation of rounding, not cohort equivalence, not an accepted primary aggregate. All original rows checked against exact upward float32 conversion envelope; all corrective rows satisfy original strict bound. No tolerance added to primary aggregator.')
(out/'paired_sensitivity.json').write_text(json.dumps(result,indent=2,allow_nan=False))
(out/'own_detector_summary.json').write_text(json.dumps(dict(recipe_changes=[dict(attack=k[0],budget=k[1],method=k[2],**c) for k,c in sorted(own.items()) if c['rejection_flips'] or c['accepted_targeted_flips']],character_union_changes=[dict(attack=k[0],budget=k[1],method=k[2],**c) for k,c in sorted(final.items()) if c['accepted_targeted_flips']],all147groups72pairedcharacters=True),indent=2))
(out/'summary.md').write_text('# Paired numerical sensitivity\n\n'+json.dumps(dict(tot),indent=2)+'\n\n'+result['interpretation']+'\n')
print('Completed fixed paired tasks10,15; compact result saved. Primary protocol unchanged.',flush=True)
