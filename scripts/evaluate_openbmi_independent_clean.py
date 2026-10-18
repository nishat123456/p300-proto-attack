"""Dedicated frozen-manifest independent runner. No candidate release accepted."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from p3a.data.openbmi_release import load_execution_release
from scripts.fit_openbmi_references import digest
from scripts.train_openbmi_source import validate_source_policy
from p3a.data.loaders.openbmi import build_trials
from p3a.data.epochs import epoch_window
from p3a.models.eegnet import EEGNet
from p3a.preproc.pipeline import Preprocessor
from p3a.defenses.raw_detector_pipeline import RawDetectorPipeline
from p3a.defenses.clean_calibration import METHODS,reject_evaluation_scores
from p3a.eval.reliability_metrics import clean_rejection_accounting,exact_binomial_interval


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','references','calibration','user-preprocessor','raw-file','schedule','out-dir','git-commit'):
        p.add_argument('--'+name,required=True)
    for name in ('protocol','manifest','release-receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--validation-integration',action='store_true',help='fixed subject19 only; never opens sealed evaluation')
    p.add_argument('--subject',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    calibration=json.loads(Path(a.calibration).read_text())
    if calibration['subject']!=a.subject or calibration['seed']!=a.seed:raise ValueError('calibration identity mismatch')
    for name,path in [('checkpoint',a.checkpoint),('reference',a.references),('raw',a.raw_file),('schedule',a.schedule)]:
        if calibration[name+'_sha256']!=digest(path):raise ValueError('calibration input hash mismatch: '+name)
    manifest,protocol_hash,manifest_hash=load_execution_release(a.protocol,a.manifest,a.release_receipt,a.subject,a.seed,digest(a.raw_file),validation_integration=a.validation_integration)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=checkpoint['config'];validate_source_policy(c)
    if checkpoint['seed']!=a.seed:raise ValueError('checkpoint seed mismatch')
    state=torch.load(a.user_preprocessor,map_location='cpu',weights_only=False)
    if state['seed']!=a.seed or state['subject']!=a.subject or state['normalization_ids']!=calibration['normalization_ids']:
        raise ValueError('normalization artifact identity mismatch')
    pp=c['preprocessing'];pre=Preprocessor(62,pp['raw_fs'],band=tuple(pp['band_hz']),target_fs=pp['target_fs'],car=pp['car'])
    pre.load_state_dict(state['preprocessor'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');pre.to(device)
    model=EEGNet(62,epoch_window(pre.fs_out),**{k:v for k,v in c['decoder'].items() if k!='architecture'}).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    with np.load(a.references,allow_pickle=False) as refs:
        features=refs['features'].copy();labels=refs['labels'].copy();means=refs['means'].copy();precision=refs['precision'].copy()
    schedule=loadmat(a.schedule,simplify_cells=True)['rc_order'][:5]
    data=loadmat(a.raw_file,simplify_cells=True,variable_names=['EEG_ERP_test'])['EEG_ERP_test']
    trials=build_trials(data,schedule,'PATTERN_RECOGNITION_MACHINE_LEARNING',a.subject,1,'test')
    ids=[t.trial_id for t in trials]
    if len(ids)!=36 or ids!=calibration['evaluation_ids']:raise ValueError('exact36 independent clean evaluation IDs required')
    if ids!=manifest['partition']['evaluation']:raise ValueError('loaded trial identities differ from frozen manifest')
    if calibration.get('protocol_sha256')!=protocol_hash or calibration.get('manifest_sha256')!=manifest_hash:raise ValueError('calibration release binding mismatch')
    from p3a.eval.reliability_cost import marker_duration_seconds
    durations=[marker_duration_seconds(t.onsets,t.fs) for t in trials]
    scores={m:[] for m in METHODS};correct=[];predictions=[]
    for trial in trials:
        pipeline=RawDetectorPipeline(pre,model,torch.as_tensor(trial.onsets,device=device).long().unsqueeze(0),
            torch.as_tensor(trial.flash_groups,device=device).bool().unsqueeze(0),features,means,precision,20,100,a.seed,.001,
            reference_labels=labels,kde_bandwidth=.5).to(device)
        raw=torch.as_tensor(trial.eeg,device=device).unsqueeze(0)
        with torch.no_grad():predicted=int(pipeline.logits(raw).argmax(dim=1).item())
        predictions.append(predicted);correct.append(predicted==trial.layout.index(trial.target_symbol))
        for method in METHODS:scores[method].append(float(pipeline.score(raw,method).detach().cpu().item()))
        print('validation clean decision',trial.trial_id,flush=True)
    rows=[];summaries=[]
    for method in METHODS:
        for alpha in (.01,.05,.1):
            rejected=reject_evaluation_scores(calibration,method,alpha,scores[method],ids,100)
            metrics=clean_rejection_accounting(np.asarray(correct,dtype=bool),rejected)
            metrics['far_iid_ci95']=exact_binomial_interval(metrics['n_rejected'],36)
            summaries.append(dict(seed=a.seed,subject=a.subject,method=method,nominal_far=alpha,**metrics))
            for i,identity in enumerate(ids):
                rows.append(dict(seed=a.seed,subject=a.subject,trial_id=identity,method=method,nominal_far=alpha,
                    anomaly=scores[method][i],threshold=calibration['methods'][method]['thresholds'][str(alpha)],
                    predicted_symbol=trials[i].layout.symbols[predictions[i]],intended_symbol=trials[i].target_symbol,
                    correct=bool(correct[i]),rejected=bool(rejected[i]),scope='frozen independent clean evaluation, no method selection'))
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/f'validation_clean_s{a.subject:02d}_seed{a.seed}.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/f'validation_clean_s{a.subject:02d}_seed{a.seed}.json').write_text(json.dumps(dict(seed=a.seed,subject=a.subject,
        git_commit=a.git_commit,runner_sha256=digest(__file__),checkpoint_sha256=digest(a.checkpoint),calibration_sha256=digest(a.calibration),
        reference_sha256=digest(a.references),raw_sha256=digest(a.raw_file),user_preprocessor_sha256=digest(a.user_preprocessor),
        summaries=summaries,protocol_sha256=protocol_hash,manifest_sha256=manifest_hash,marker_duration_seconds=durations,scope='independent frozen clean evaluation, not attack TPR',
        limitations='36 serial decisions; IID intervals not per-user guarantees, no target threshold tuning or bestseed selection'),indent=2))
if __name__=='__main__':main()
