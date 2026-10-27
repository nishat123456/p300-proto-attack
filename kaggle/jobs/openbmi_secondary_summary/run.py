"""Kaggle-only supplementary reporting of existing aggregate, no new evaluation."""
import json,hashlib,math,itertools,unittest
from pathlib import Path
import numpy as np
EXPECTED='e11fb26676c22c548074ecbf8113f4aa427a815df2723c383effb277cf31e9ce'
METHODS=('confidence','entropy','bu','lid','md','md_adjusted','kde')
SEEDS=(20261013,20261014,20261015)
SUBJECTS=tuple(x for x in range(37,55) if x!=52)
def summarize(values):
 if not values:return None
 a=np.array(values,dtype=float)
 if not np.isfinite(a).all() or ((a<0)|(a>1)).any():raise ValueError('invalid rate')
 rng=np.random.default_rng(20261013)
 means=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(axis=1)
 return dict(mean=float(a.mean()),ci95=list(map(float,np.percentile(means,[2.5,97.5]))),n_subjects=len(a),n_resamples=2000,seed=20261013)
def grouped(rows,keys,metric):
 groups={}
 for r in rows:groups.setdefault(tuple(r[k] for k in keys),[]).append(r)
 result=[]
 for k,rs in sorted(groups.items()):
  ids={(r['subject'],r['seed']) for r in rs}
  assert len(rs)==51 and ids==set(itertools.product(SUBJECTS,SEEDS))
  users={};undefined=[]
  for s in SUBJECTS:
   sr=sorted((r for r in rs if r['subject']==s),key=lambda r:r['seed'])
   vals=[r[metric] for r in sr]
   if any(v is None for v in vals):undefined.append(s)
   else:users[str(s)]=float(np.mean(vals))
  # Main all-user summary unavailable if any subject has an undefined seed.
  result.append(dict(zip(keys,k),metric=metric,subject_seed_averages=users,undefined_subjects=undefined,all17_summary=None if undefined else summarize(list(users.values())),defined_only_descriptive=None if not undefined else summarize(list(users.values())),per_seed=rs))
 return result
class Guards(unittest.TestCase):
 def test_zero(self):self.assertEqual(summarize([0,0])['mean'],0)
 def test_one(self):self.assertEqual(summarize([1,1])['mean'],1)
 def test_nonfinite(self):self.assertRaises(ValueError,summarize,[float('nan')])
 def test_bounds(self):self.assertRaises(ValueError,summarize,[1.1])
 def test_missing(self):self.assertIsNone(summarize([]))
if __name__=='__main__':
 assert Path('/kaggle/working').is_dir(),'cloud only'
 suite=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards));assert suite.wasSuccessful()
 paths=list(Path('/kaggle/input').rglob('reliability_aggregate.json'))
 matches=[p for p in paths if hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED];assert len(matches)==1
 p=matches[0];data=json.loads(p.read_text());assert data['original_strict_contract_satisfied'] is False
 assert tuple(data['subjects'])==SUBJECTS and tuple(data['seeds'])==SEEDS
 clean=data['clean_per_seed'];attack=data['per_seed']
 for r in clean:
  r['accepted_denominator']=r['n_characters']-r['n_rejected']
  v=r['conditional_accepted_accuracy']
  assert (v is None)==(r['accepted_denominator']==0)
 for r in attack:
  assert r['n_attempts']==36
  assert (r['conditional_tpr'] is None)==(r['conditional_tpr_denominator']==0)
 result=dict(contract=data['numerical_analysis_contract'],original_strict_contract_satisfied=False,input_sha256=EXPECTED,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),guard_tests=suite.testsRun,seed=20261013,clean_conditional=grouped(clean,['nominal_far','method'],'conditional_accepted_accuracy'),clean_decoder=grouped(clean,['nominal_far','method'],'clean_accuracy'),conditional_tpr=grouped(attack,['attack','budget','method'],'conditional_tpr'),nonadaptive_auc=grouped([r for r in attack if r['attack']!='detector_aware'],['attack','budget','method'],'detector_unaware_auc_all_attempts'),limitations='Secondary post-execution descriptive report; equal-user after three-seed averaging; undefined seeds retained, all17 summary withheld when any undefined; defined-only summaries explicitly conditional; pointwise bootstrap not familywise or serial guarantees; no causal rounding equivalence.')
 out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
 (out/'secondary_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 lines=['# Secondary descriptive metrics','',result['limitations'],'','All original strict failures remain disclosed. No new attacks, model fitting or thresholds.','']
 for section in ['clean_conditional','clean_decoder','conditional_tpr','nonadaptive_auc']:
  lines+=['## '+section,'','|Group|All17 mean [pointwise95% CI]|Undefined participants|','|---|---|---|']
  for r in result[section]:
   label=' / '.join(str(r[k]) for k in ['nominal_far','attack','budget','method'] if k in r)
   x=r['all17_summary'];v='unavailable' if x is None else f"{x['mean']:.6f} [{x['ci95'][0]:.6f},{x['ci95'][1]:.6f}]"
   lines.append(f"|{label}|{v}|{r['undefined_subjects']}|")
  lines.append('')
 (out/'secondary_summary.md').write_text('\n'.join(lines))
 print('Complete secondary summary; inspect undefined cases before interpretation.')
