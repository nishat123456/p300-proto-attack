"""Cloud-only audit of fixed development outputs, no model selection."""
import hashlib,json,math
from pathlib import Path
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
methods=['confidence','entropy','BU','LID','MD','MD_adjusted','KDE']
reports=[]
for seed in (20261013,20261014,20261015):
    paths=list(Path('/kaggle/input').rglob(f'attack_audit_seed{seed}.json'))
    assert len(paths)==1, paths
    p=paths[0];d=json.loads(p.read_text());rows=d['rows']
    assert d['seed']==seed and d['subject']==19 and len(rows)==76
    assert d['gradient_draws']==30 and d['verification_draws']==100
    def finite(x):
        if isinstance(x,float):assert math.isfinite(x)
        elif isinstance(x,dict):
            for v in x.values():finite(v)
        elif isinstance(x,list):
            for v in x:finite(v)
    finite(d)
    for r in rows:
        assert 0<=r['applied_linf']<=r['raw_epsilon']
        assert r['targeted_success']==(r['predicted']==d['target'])
        r['targeted_and_accepted']={m:r['targeted_success'] and not reject for m,reject in r['rejected_at_nominal5'].items()}
    pairs=[]
    for t in d['traces']:
        tr=t['trace'];s=t['settings']
        assert len(tr)==s['restarts']*(s['steps']//20+1)
        objectives=[x['best_objective'][0] for x in tr]
        assert all(b<=a for a,b in zip(objectives,objectives[1:]))
        t['first_best_objective']=objectives[0];t['last_best_objective']=objectives[-1]
        del t['trace']
    for r in rows:
        s=r['settings']
        if r['attack']=='detector_aware' and s['steps']==200:
            longer=[v for v in rows if v['attack']=='detector_aware' and v['settings']['steps']==400 and v['settings']['detector']==s['detector'] and v['settings']['weight']==s['weight']]
            assert len(longer)==1
            v=longer[0]
            pairs.append(dict(detector=s['detector'],weight=s['weight'],short_targeted_success=r['targeted_success'],long_targeted_success=v['targeted_success'],short_accepted=r['targeted_and_accepted'],long_accepted=v['targeted_and_accepted'],short_seconds=r['optimization_seconds'],long_seconds=v['optimization_seconds']))
    reports.append(dict(seed=seed,input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows,traces=d['traces'],convergence_pairs=pairs,zero_gradient_methods=sorted({r['settings']['detector'] for r in rows if r['attack']=='detector_aware' and r['settings']['gradient']['zero']})))
result=dict(scope='One fixed development character, three model seeds. Not population TPR, independent confirmation or robustness proof.',checks='All76 rows per seed finite, actual budgets respected, target identities consistent; trace lengths and best-objective monotonicity checked.',reports=reports)
(out/'attack_summary.json').write_text(json.dumps(result,indent=2))
(out/'attack_summary.md').write_text('# Development optimization audit\n\n'+result['scope']+'\n\n'+result['checks']+'\n\nAll settings and paired convergence evidence are preserved in attack_summary.json. No detector or seed selected. Independent gate remains closed.\n')
print(result['checks'],flush=True)
