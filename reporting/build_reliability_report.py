"""Cloud-only presentation of frozen aggregate outputs; no detector fitting."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['confidence','entropy','bu','lid','md','md_adjusted','kde']
ATTACKS=['FGSM','PGD','Gaussian_control','source_template','detector_aware']
BUDGETS=[.05,.1,.2,.3,.5]
SEEDS=[20261013,20261014,20261015]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(row,development):
    result=row.get('subject_mean_uncertainty')
    if result is None:
        if development:return None
        raise ValueError('population uncertainty missing')
    ci=result['ci95']
    if not development and (ci is None or len(ci)!=2):raise ValueError('population interval required')
    return ci

def validate(data,protocol_hash,development):
    subjects=[19] if development else [s for s in range(37,55) if s!=52]
    if data['subjects']!=subjects or data['seeds']!=SEEDS:raise ValueError('complete fixed subject/seed coverage required')
    if data['protocol_sha256']!=protocol_hash:raise ValueError('aggregate protocol mismatch')
    groups={(a,b,m) for a in ATTACKS for b in ([.2] if a=='detector_aware' else BUDGETS) for m in METHODS}
    records={}
    for row in data['population']:
        key=(row['attack'],row['budget'],row['method'])
        if key in records:raise ValueError('duplicate population summary')
        records[key]=row
        if set(map(int,row['subject_seed_averages']))!=set(subjects):raise ValueError('missing subject summary')
        values=list(row['subject_seed_averages'].values())+[row['mean_defended_targeted_success']]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in values):raise ValueError('invalid rate')
        interval(row,development)
    if set(records)!=groups:raise ValueError('all methods/budgets/attacks required')
    rows=data['per_seed']
    identities={(r['subject'],r['seed'],r['attack'],r['budget'],r['method']) for r in rows}
    expected={(s,seed,*group) for s in subjects for seed in SEEDS for group in groups}
    if identities!=expected or len(rows)!=len(expected):raise ValueError('complete nonduplicate per-seed rows required')
    if any(r['n_attempts']!=36 for r in rows):raise ValueError('original36 attempt denominators required')
    clean={}
    for row in data['clean_population']:
        key=(row['nominal_far'],row['method'],row['metric'])
        if key in clean:raise ValueError('duplicate clean summary')
        clean[key]=row
        if set(map(int,row['subject_seed_averages']))!=set(subjects):raise ValueError('missing clean subject summary')
        if any(not math.isfinite(v) or not 0<=v<=1 for v in [row['mean'],*row['subject_seed_averages'].values()]):raise ValueError('invalid clean rate')
        interval(row,development)
    expected_clean={(alpha,m,metric) for alpha in (.01,.05,.1) for m in METHODS for metric in ('observed_far','clean_accuracy','accepted_correct_rate','accepted_wrong_rate')}
    if set(clean)!=expected_clean:raise ValueError('all clean nominal/method/metric summaries required')
    return subjects,records,clean

def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('report processing must run on Kaggle')
    p=argparse.ArgumentParser()
    for name in ('aggregate','protocol','out-dir','git-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--development-only',action='store_true');a=p.parse_args()
    data=json.loads(Path(a.aggregate).read_text());subjects,records,clean=validate(data,digest(a.protocol),a.development_only)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    scope='DEVELOPMENT ONLY: fixed already-inspected subject19' if a.development_only else 'Independent available-case17 users;18 planned,52 unavailable'
    def rate(value,ci):
        text=f'{100*value:.2f}%'
        return text+(' (population CI unavailable)' if ci is None else f' [{100*ci[0]:.2f}, {100*ci[1]:.2f}]')
    lines=['# Character-level reliability report','',scope,'',
      'All rates retain original character denominators. Model seeds are averaged within each user before equal-user means. Intervals are pointwise subject-bootstrap descriptions under exchangeability, not per-user guarantees. Adaptive rows are separate attacks against each detector, not simultaneous evasion or comparative efficacy.','',
      '## Clean decisions at nominal5%','', '|Method|Observed rejection mean [95% CI]|Accepted correct / all clean|Accepted wrong / all clean|','|---|---|---|---|']
    for m in METHODS:
        far=clean[(.05,m,'observed_far')];correct=clean[(.05,m,'accepted_correct_rate')];wrong=clean[(.05,m,'accepted_wrong_rate')]
        lines.append('|'+m+'|'+rate(far['mean'],interval(far,a.development_only))+'|'+rate(correct['mean'],interval(correct,a.development_only))+'|'+rate(wrong['mean'],interval(wrong,a.development_only))+'|')
    lines+=['','## All methods and declared budgets','','|Attack|Budget|Detector|Defended targeted / all attempts, mean [95% CI]|','|---|---|---|---|']
    for key,row in sorted(records.items()):lines.append(f'|{key[0]}|{key[1]:.2f}|{key[2]}|'+rate(row['mean_defended_targeted_success'],interval(row,a.development_only))+'|')
    plt.rcParams.update({'font.size':9,'savefig.dpi':180})
    fig,ax=plt.subplots(figsize=(9,max(3,len(subjects)*.28+1.6)))
    values=np.array([[clean[(.05,m,'observed_far')]['subject_seed_averages'][str(s)]*100 for m in METHODS] for s in subjects])
    im=ax.imshow(values,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.set_xticks(range(7),METHODS,rotation=30,ha='right');ax.set_yticks(range(len(subjects)),subjects)
    ax.set_ylabel('Participant');ax.set_title(scope+'\nObserved clean character rejection, nominal5%')
    fig.colorbar(im,ax=ax,label='Rejected / all clean (%)');fig.tight_layout();fig.savefig(out/'clean_rejection.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    values=np.array([[records[(attack,.2,m)]['mean_defended_targeted_success']*100 for attack in ATTACKS] for m in METHODS])
    im=ax.imshow(values,aspect='auto',vmin=0,vmax=100,cmap='magma')
    ax.set_xticks(range(5),ATTACKS,rotation=25,ha='right');ax.set_yticks(range(7),METHODS)
    ax.set_title(scope+'\nTargeted manipulation accepted / all attempts, budget0.20')
    fig.colorbar(im,ax=ax,label='Accepted targeted manipulation (%)');fig.tight_layout();fig.savefig(out/'defended_manipulation.png');plt.close(fig)
    lines+=['','## Figure interpretation','','![Observed clean rejection](clean_rejection.png)','','![Accepted manipulation](defended_manipulation.png)','',
      'Color scales are fixed0-100%. Tables carry pointwise population intervals; colors do not establish significance. Compare detector-specific adaptive rows only as stress tests. The fixed longer run covers the first character per user/seed, not every character. No winner, robust-defense or journal-readiness verdict is generated.','',
      '## Limitations','','Twenty calibration and36 test characters give coarse user estimates. Serial dependence, phrase/phase shift,17 conditional available cases, recorded-marker timing and fixed adaptive recipe limit inference. Rejection is not clinical throughput. BNCI2015-003 remains locked.','']
    (out/'reliability_report.md').write_text('\n'.join(lines))
    proof=dict(scope=scope,development_only=a.development_only,subjects=subjects,seeds=SEEDS,git_commit=a.git_commit,reporter_sha256=digest(__file__),aggregate_sha256=digest(a.aggregate),protocol_sha256=digest(a.protocol),files={p.name:digest(p) for p in out.iterdir()},visual_review='pending; generation alone does not establish figure layout quality')
    (out/'report_provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__':main()
