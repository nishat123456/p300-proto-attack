"""Cloud arithmetic only; not a measured runtime guarantee."""
import json,hashlib
from pathlib import Path
out=Path('/kaggle/working/out');out.mkdir(exist_ok=True)
paths=list(Path('/kaggle/input').rglob('attack_summary.json'));assert len(paths)==1
p=paths[0];d=json.loads(p.read_text())
assert len(d['reports'])==3
estimates=[]
for report in d['reports']:
    rows=report['rows']
    baseline=sum(r['optimization_seconds'] for r in rows if r['attack']!='detector_aware')
    short=sum(r['optimization_seconds'] for r in rows if r['attack']=='detector_aware' and r['settings']['steps']==200)
    longer=sum(r['optimization_seconds'] for r in rows if r['attack']=='detector_aware' and r['settings']['steps']==400)
    estimates.append(dict(seed=report['seed'],baseline_seconds=baseline,adaptive_short_seconds_one_budget=short,adaptive_long_seconds_primary=longer,estimated_seconds_per_character_all_budgets=baseline+5*short+longer))
total_seconds=17*36*sum(x['estimated_seconds_per_character_all_budgets'] for x in estimates)
result=dict(scope='Planning extrapolation from ONE development character; excludes detector verification, calibration, downloads, checkpoint I/O and failures. Not measured benchmark runtime.',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),subjects=17,characters_per_subject=36,seeds=3,baseline_settings_per_character_seed=20,adaptive_short_settings_per_character_seed=140,adaptive_long_settings_per_character_seed=28,estimates=estimates,estimated_serial_gpu_hours=total_seconds/3600,estimated_serial_gpu_days=total_seconds/86400,settings_all_subjects_seeds=17*36*3*(20+140+28),gate='No automatic scope reduction or performance unlock')
(out/'workload.json').write_text(json.dumps(result,indent=2))
(out/'workload.md').write_text('# Prospective workload accounting\n\n'+result['scope']+'\n\nSettings: '+str(result['settings_all_subjects_seeds'])+'\n\nEstimated serial GPU hours: '+str(result['estimated_serial_gpu_hours'])+'\n\nEstimated serial GPU days: '+str(result['estimated_serial_gpu_days'])+'\n\nThis is a resource-feasibility diagnostic, not permission to choose a favorable subset or discard budgets.\n')
print(result['scope'],result['estimated_serial_gpu_hours'])
