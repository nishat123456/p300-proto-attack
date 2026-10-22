"""Cloud-only complete-case reliability aggregation, all settings preserved."""
import argparse,json,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.fit_openbmi_references import digest
from p3a.defenses.clean_calibration import METHODS
from p3a.eval.reliability_metrics import paired_subject_bootstrap,exact_binomial_interval,subject_mean_bootstrap


def main():
    p=argparse.ArgumentParser()
    for name in ('root','protocol','out-dir','git-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--validation-integration',action='store_true');a=p.parse_args()
    root=Path(a.root);protocol=json.loads(Path(a.protocol).read_text());ph=digest(a.protocol)
    subjects=[19] if a.validation_integration else [s for s in protocol['subject_roles']['evaluation'] if s!=52]
    seeds=protocol['seeds'];by_subject_seed={};per_seed=[];clean_rows=[];files={}
    for subject in subjects:
        for seed in seeds:
            candidates=list(root.rglob(f'validation_clean_s{subject:02d}_seed{seed}.json'))
            if len(candidates)!=1:raise ValueError('missing or duplicate clean result')
            cp=candidates[0];clean=json.loads(cp.read_text());files[str(cp)]=digest(cp)
            if clean['subject']!=subject or clean['seed']!=seed:raise ValueError('clean identity mismatch')
            if clean['protocol_sha256']!=ph:raise ValueError('clean protocol mismatch')
            selected=[r for r in clean['summaries'] if r['nominal_far']==.05]
            if len(selected)!=7 or {r['method'] for r in selected}!=set(METHODS):raise ValueError('all seven clean methods required')
            if len(clean['summaries'])!=21 or any(r['n_characters']!=36 for r in clean['summaries']):raise ValueError('all declared nominal clean outcomes required')
            clean_rows.extend(clean['summaries'])
            with cp.with_suffix('.csv').open() as handle:
                original=[r for r in csv.DictReader(handle) if float(r['nominal_far'])==.05]
            clean_scores={m:[float(r['anomaly']) for r in original if r['method']==m] for m in METHODS}
            if any(len(x)!=36 for x in clean_scores.values()):raise ValueError('36 original clean scores required per detector')
            records={};grouped=defaultdict(list);anomalies=defaultdict(list)
            for path in root.rglob(f'attack_audit_seed{seed}.json'):
                data=json.loads(path.read_text())
                if data['subject']!=subject:continue
                if data['seed']!=seed:raise ValueError('attack model seed mismatch')
                if data['protocol_sha256']!=ph or data['manifest_sha256']!=clean['manifest_sha256']:raise ValueError('attack protocol/manifest mismatch')
                identity=data['trial_id']
                if identity in records:raise ValueError('duplicate character shard')
                records[identity]=data;files[str(path)]=digest(path)
                expected=34 if identity.endswith(':000') else 27
                if len(data['rows'])!=expected:raise ValueError('incomplete declared setting count')
                local=defaultdict(list)
                for row in data['rows']:
                    if not 0<=row['applied_linf']<=row['raw_epsilon']:raise ValueError('actual budget failure')
                    attack=row['attack'];budget=row['budget']
                    methods=[row['settings']['detector']] if attack=='detector_aware' else METHODS
                    for method in methods:
                        key=(attack,budget,method)
                        local[key].append(row)
                for key,candidates in local.items():
                    method=key[2]
                    required=2 if key[0]=='detector_aware' and identity.endswith(':000') else 1
                    if len(candidates)!=required:raise ValueError('duplicate or missing declared recipe setting')
                    targeted=any(r['targeted_success'] for r in candidates)
                    accepted=any(r['targeted_success'] and not r['rejected_at_nominal5'][method] for r in candidates)
                    grouped[key].append((targeted,accepted))
                    # AUC reported only for a single nonadaptive perturbation per character.
                    if key[0]!='detector_aware':anomalies[key].append(candidates[0]['scores'][method])
            expected_ids={f'openbmi_candidate:s{subject:02d}_session1:test:{i:03d}' for i in range(36)}
            if set(records)!=expected_ids:raise ValueError('all36 original character shards required')
            expected_groups=4*5*7+7
            if len(grouped)!=expected_groups:raise ValueError('missing attack/budget/method group')
            for (attack,budget,method),outcomes in sorted(grouped.items()):
                if len(outcomes)!=36:raise ValueError('original36 attempt denominator required')
                success=sum(t for t,_ in outcomes);accepted=sum(v for _,v in outcomes)
                auc=None
                if attack!='detector_aware':
                    scores=clean_scores[method]+anomalies[(attack,budget,method)]
                    if not np.isfinite(scores).all():raise ValueError('nonfinite AUC input')
                    auc=float(roc_auc_score([0]*36+[1]*36,scores))
                per_seed.append(dict(subject=subject,seed=seed,attack=attack,budget=budget,method=method,n_attempts=36,n_targeted=success,n_defended_targeted=accepted,targeted_success_rate=success/36,defended_targeted_success_rate=accepted/36,conditional_tpr=None if not success else (success-accepted)/success,conditional_tpr_denominator=success,defended_targeted_iid_ci95=exact_binomial_interval(accepted,36),detector_unaware_auc_all_attempts=auc))
    # Average model seeds within each subject before equal-subject summaries.
    population=[];comparisons=[];user_values=defaultdict(dict)
    for row in per_seed:
        key=(row['attack'],row['budget'],row['method'])
        user_values[key].setdefault(row['subject'],[]).append(row['defended_targeted_success_rate'])
    for key,users in sorted(user_values.items()):
        if set(users)!=set(subjects) or any(len(v)!=len(seeds) for v in users.values()):raise ValueError('incomplete subject/seed coverage')
        means={s:float(np.mean(users[s])) for s in subjects}
        population.append(dict(attack=key[0],budget=key[1],method=key[2],mean_defended_targeted_success=float(np.mean(list(means.values()))),subject_seed_averages=means,subject_mean_uncertainty=subject_mean_bootstrap([means[s] for s in subjects],a.seed,2000)))
        comparator=user_values.get((key[0],key[1],'confidence'))
        # Each adaptive attack targets a different defense; no paired comparative efficacy claim.
        if key[2]!='confidence' and key[0]!='detector_aware' and len(subjects)>1:
            comparisons.append(dict(attack=key[0],budget=key[1],method=key[2],versus='confidence',**paired_subject_bootstrap([means[s] for s in subjects],[np.mean(comparator[s]) for s in subjects],a.seed,2000)))
    clean_population=[]
    for alpha in (.01,.05,.1):
        for method in METHODS:
            rows=[r for r in clean_rows if r['method']==method and r['nominal_far']==alpha]
            if len(rows)!=len(subjects)*len(seeds):raise ValueError('incomplete clean nominal/seed coverage')
            for metric in ('observed_far','clean_accuracy','accepted_correct_rate','accepted_wrong_rate'):
                values={s:float(np.mean([r[metric] for r in rows if r['subject']==s])) for s in subjects}
                clean_population.append(dict(nominal_far=alpha,method=method,metric=metric,mean=float(np.mean(list(values.values()))),subject_seed_averages=values,subject_mean_uncertainty=subject_mean_bootstrap([values[s] for s in subjects],a.seed,2000)))
    result=dict(seed=a.seed,git_commit=a.git_commit,runner_sha256=digest(__file__),protocol_sha256=ph,subjects=subjects,seeds=seeds,per_seed=per_seed,clean_per_seed=clean_rows,clean_population=clean_population,population=population,paired_descriptive_comparisons=comparisons,input_sha256=files,scope='validation-only aggregation integration' if a.validation_integration else '17 eligible subjects conditional available-case;18 in evaluation flow with52 unavailable',limitations='serial decisions; IID intervals descriptive only;pointwise bootstrap not familywise significance;fixed recipe not worst-case;adaptive union only same-character declared candidates')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'reliability_aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
