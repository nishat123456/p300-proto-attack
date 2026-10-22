"""Cloud-only display of recorded cost fields, no detector fitting or timing."""
import argparse,hashlib,json,math
from pathlib import Path
METHODS=['confidence','entropy','bu','lid','md','md_adjusted','kde']
SEEDS=[20261013,20261014,20261015]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(clean,timing,ph,development):
    subjects=[19] if development else [s for s in range(37,55) if s!=52]
    expected={(s,seed) for s in subjects for seed in SEEDS}
    if len(clean)!=len(expected) or {(d['subject'],d['seed']) for d in clean}!=expected:raise ValueError('exact complete clean subject/seed coverage required')
    if len(timing)!=3 or {d['seed'] for d in timing}!=set(SEEDS):raise ValueError('all three source timing seeds required')
    lookup={d['seed']:d for d in timing}
    for d in timing:
        if d['trial_id']!='openbmi_candidate:s01_session1:train:000' or not d.get('hardware'):raise ValueError('fixed source query and hardware required')
        if set(d['methods'])!=set(METHODS):raise ValueError('all seven measured methods required')
        for row in d['methods'].values():
            if row['warmup']!=5 or row['repeats']!=30 or len(row['seconds'])!=30:raise ValueError('declared timing repetitions required')
            values=row['seconds']+[row['median_seconds'],row['p95_seconds']]
            if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('positive finite timing required')
    for d in clean:
        if d['protocol_sha256']!=ph:raise ValueError('clean protocol mismatch')
        t=lookup[d['seed']]
        if d['checkpoint_sha256']!=t['checkpoint_sha256'] or d['reference_sha256']!=t['reference_sha256']:raise ValueError('timing model/reference mismatch')
        durations=d['marker_duration_seconds']
        if len(durations)!=36 or any(not math.isfinite(x) or x<=0 for x in durations):raise ValueError('all36 positive marker proxies required')
        rows=d['summaries'];keys={(r['nominal_far'],r['method']) for r in rows}
        if len(rows)!=21 or keys!={(a,m) for a in [.01,.05,.1] for m in METHODS}:raise ValueError('all clean nominal methods required')
        for r in rows:
            n=r['n_rejected'];f=r['observed_far'];retry=r['iid_retry_overhead']
            if r['subject']!=d['subject'] or r['seed']!=d['seed'] or r['n_characters']!=36 or type(n)!=int or not 0<=n<=36 or not math.isclose(f,n/36,abs_tol=1e-12):raise ValueError('original clean count mismatch')
            if n==36:
                if retry is not None:raise ValueError('all-rejected retry must be unavailable')
            elif retry is None or not math.isfinite(retry) or not math.isclose(retry,1/(1-f)-1,abs_tol=1e-10):raise ValueError('stored IID retry mismatch')
    return subjects

def main():
    p=argparse.ArgumentParser()
    for key in ['clean-root','timing-root','protocol','out-dir','git-commit']:p.add_argument('--'+key,required=True)
    p.add_argument('--development-only',action='store_true');a=p.parse_args()
    if not Path.cwd().is_relative_to('/kaggle/working'):raise RuntimeError('scientific reporting runs on Kaggle only')
    subjects=[19] if a.development_only else [s for s in range(37,55) if s!=52]
    paths=[path for s in subjects for seed in SEEDS for path in Path(a.clean_root).rglob(f'validation_clean_s{s:02d}_seed{seed}.json')]
    times=[path for seed in SEEDS for path in Path(a.timing_root).rglob(f'source_cost_seed{seed}.json')]
    clean=[json.loads(path.read_text()) for path in paths];timing=[json.loads(path.read_text()) for path in times]
    validate(clean,timing,digest(a.protocol),a.development_only)
    scope='DEVELOPMENT participant19 only, no population cost inference' if a.development_only else 'Independent17 available participants,18 in flow with52 unavailable'
    lines=['# Recorded computation and rejection cost','',scope,'','## Fixed source-query computation','','|Seed|Hardware|Method|Median seconds|P95 seconds|','|---|---|---|---|---|']
    for t in sorted(timing,key=lambda d:d['seed']):
        for m in METHODS:
            r=t['methods'][m];lines.append(f"|{t['seed']}|{t['hardware']}|{m}|{r['median_seconds']:.6f}|{r['p95_seconds']:.6f}|")
    lines+=['','Timing is the synchronized batch1 source query with five warmups/30 repeats. It is not measured independent-user latency or clinical throughput. Timing-only reference identity exclusion is preserved in the input provenance; deployment references are unchanged.','','## All declared clean operating points','','|Participant|Seed|Nominal|Method|Rejected /36|Stored ideal IID extra attempts per accepted decision|','|---|---|---|---|---|---|']
    for d in sorted(clean,key=lambda d:(d['subject'],d['seed'])):
        for r in d['summaries']:
            v=r['iid_retry_overhead'];value='unavailable: all rejected' if v is None else f'{v:.6f}'
            lines.append(f"|{d['subject']}|{d['seed']}|{r['nominal_far']}|{r['method']}|{r['n_rejected']}/36|{value}|")
    lines+=['','## Recorded marker proxy','','|Participant|Seed|First-to-last marker plus0.8s, minimum seconds|Maximum seconds|','|---|---|---|---|']
    for d in sorted(clean,key=lambda d:(d['subject'],d['seed'])):
        v=d['marker_duration_seconds'];lines.append(f"|{d['subject']}|{d['seed']}|{min(v):.6f}|{max(v):.6f}|")
    lines+=['','All36 marker proxies per participant/seed remain in the bound clean JSON. These extrema are descriptive, not uncertainty intervals. Retry overhead is a stored IID sensitivity expression, not measured repeated spelling, correction/backspace cost or a serial-dependence guarantee. No product of average timing and average retry is presented as observed latency. Participant uncertainty for primary reliability rates is reported separately; this per-seed cost table does not infer a population retry effect. No method or seed is selected.','']
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'cost_report.md').write_text('\n'.join(lines))
    (out/'cost_report_provenance.json').write_text(json.dumps(dict(scope=scope,subjects=subjects,seeds=SEEDS,git_commit=a.git_commit,reporter_sha256=digest(__file__),protocol_sha256=digest(a.protocol),input_sha256={str(path):digest(path) for path in paths+times},limitations='descriptive stored cost fields; no new timing, population retry CI or clinical throughput'),indent=2))
if __name__=='__main__':main()
