"""Manifest-bound independent attack shard, one declared character per invocation."""
import argparse,json,csv,time,hashlib
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
    for name in ('checkpoint','references','calibration','user-preprocessor','raw-file','schedule','out-dir','git-commit','template','reference-json'):
        p.add_argument('--'+name,required=True)
    for name in ('protocol','manifest','release-receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--trial-index',type=int,required=True)
    p.add_argument('--subject',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    if not 0<=a.trial_index<36:raise ValueError('fixed manifest character index0..35 required')
    manifest,protocol_hash,manifest_hash=load_execution_release(a.protocol,a.manifest,a.release_receipt,a.subject,a.seed,digest(a.raw_file))
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    calibration=json.loads(Path(a.calibration).read_text())
    if calibration['subject']!=a.subject or calibration['seed']!=a.seed:raise ValueError('calibration identity mismatch')
    for name,path in [('checkpoint',a.checkpoint),('reference',a.references),('raw',a.raw_file),('schedule',a.schedule)]:
        if calibration[name+'_sha256']!=digest(path):raise ValueError('calibration input hash mismatch: '+name)
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
    from p3a.attacks.adaptive_evasion import defense_aware_pgd,project_applied_delta
    from p3a.attacks.generators import place_template_at_onsets,linf_rescale
    # Exact character index supplied by a deterministic manifest task, never outcome selection.
    trial=trials[a.trial_index];raw=torch.as_tensor(trial.eeg,device=device).unsqueeze(0)
    on=torch.as_tensor(trial.onsets,device=device).long().unsqueeze(0)
    groups=torch.as_tensor(trial.flash_groups,device=device).bool().unsqueeze(0)
    intended=trial.layout.index(trial.target_symbol)
    target_row=manifest['targets'][a.trial_index]
    if target_row['trial_id']!=trial.trial_id or target_row['intended']!=intended:raise ValueError('frozen target identity mismatch')
    target_index=target_row['target'];target=torch.tensor([target_index],device=device)
    if ids!=manifest['partition']['evaluation']:raise ValueError('frozen evaluation identities mismatch')
    if calibration.get('protocol_sha256')!=protocol_hash or calibration.get('manifest_sha256')!=manifest_hash:raise ValueError('calibration release binding mismatch')
    refproof=json.loads(Path(a.reference_json).read_text())
    if refproof['artifact_sha256']!=digest(a.references) or refproof['source_template']['artifact_sha256']!=digest(a.template):raise ValueError('template/reference provenance mismatch')
    template=torch.load(a.template,map_location=device,weights_only=False)['template'].unsqueeze(0)
    def pipeline(draws,seed):
        return RawDetectorPipeline(pre,model,on,groups,features,means,precision,20,draws,seed,.001,reference_labels=labels,kde_bandwidth=.5).to(device)
    eot=pipeline(30,a.seed);verify=pipeline(100,a.seed+1000000)
    # Shards are immutable once complete; accidental repeats must not overwrite evidence.
    out=Path(a.out_dir)/f's{a.subject:02d}_seed{a.seed}_trial{a.trial_index:03d}';out.mkdir(parents=True,exist_ok=True);rows=[];traces=[]
    metadata=dict(seed=a.seed,subject=a.subject,trial_id=trial.trial_id,target=target_index,intended=intended,
        target_rule='fixed SHA256 seed20261013+trialID, exclude intended, shared all models/settings',
        git_commit=a.git_commit,runner_sha256=digest(__file__),checkpoint_sha256=digest(a.checkpoint),
        reference_sha256=digest(a.references),calibration_sha256=digest(a.calibration),template_sha256=digest(a.template),
        scope='frozen independent character shard; no method/seed selection',protocol_sha256=protocol_hash,manifest_sha256=manifest_hash,
        gradient_draws=30,verification_draws=100,verification_seed=a.seed+1000000,raw_sha256=digest(a.raw_file),schedule_sha256=digest(a.schedule),user_preprocessor_sha256=digest(a.user_preprocessor))
    from p3a.eval.shard_checkpoint import load_shard,save_shard,setting_key
    checkpoint_path=out/f'attack_audit_seed{a.seed}.json'
    rows,traces=load_shard(checkpoint_path,metadata)
    completed={setting_key(r['attack'],r['budget'],r['settings']) for r in rows}
    def done(name,budget,settings):return setting_key(name,budget,settings) in completed
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize()
    def record(name,budget,delta,settings,elapsed):
        actual=(raw+delta)-raw;epsilon=budget*calibration['normalization_raw_scale']
        if not torch.isfinite(delta).all() or actual.abs().max()>epsilon:raise ValueError('applied raw budget or finite failure')
        with torch.no_grad():predicted=int(verify.logits(raw+delta).argmax(1).item())
        scores={m:float(verify.score(raw+delta,m).detach().cpu().item()) for m in METHODS}
        rejected={m:bool(scores[m]>calibration['methods'][m]['thresholds']['0.05']) for m in METHODS}
        rows.append(dict(attack=name,budget=budget,raw_epsilon=epsilon,applied_linf=float(actual.abs().max()),
            predicted=predicted,targeted_success=predicted==target_index,scores=scores,rejected_at_nominal5=rejected,
            optimization_seconds=elapsed,settings=settings))
        save_shard(checkpoint_path,metadata,rows,traces)
        completed.add(setting_key(name,budget,settings))
        print('attack path complete',a.seed,name,budget,settings,flush=True)
    for budget in (.05,.1,.2,.3,.5):
        eps=budget*calibration['normalization_raw_scale']
        if not done('FGSM',budget,dict(steps=1)):
            synchronize();start=time.perf_counter()
            z=raw.detach().clone().requires_grad_(True)
            gradient=torch.autograd.grad(torch.nn.functional.cross_entropy(eot.logits(z),target),z)[0]
            synchronize();record('FGSM',budget,project_applied_delta(raw,-eps*gradient.sign(),eps),dict(steps=1),time.perf_counter()-start)
        if not done('Gaussian_control',budget,{}):
            rng=torch.Generator(device=device).manual_seed(a.seed)
            synchronize();start=time.perf_counter();noise=torch.randn(raw.shape,device=device,generator=rng)
            delta=project_applied_delta(raw,linf_rescale(noise,eps),eps)
            synchronize();record('Gaussian_control',budget,delta,{},time.perf_counter()-start)
        if not done('source_template',budget,dict(source_updates=200)):
            synchronize();start=time.perf_counter();placed=place_template_at_onsets(template,on,groups,target,raw.shape[-1])
            delta=project_applied_delta(raw,linf_rescale(placed,eps),eps)
            synchronize();record('source_template',budget,delta,dict(source_updates=200),time.perf_counter()-start)
        if not done('PGD',budget,dict(steps=200,restarts=3)):
            synchronize();start=time.perf_counter()
            delta=defense_aware_pgd(raw,target,eot.logits,lambda z:z.flatten(1).sum(1)*0,0.,1.,eps,0.,200,3,a.seed)
            synchronize();record('PGD',budget,delta,dict(steps=200,restarts=3),time.perf_counter()-start)
    eps=.2*calibration['normalization_raw_scale']
    for method in METHODS:
        threshold=calibration['methods'][method]['thresholds']['0.05'];scale=calibration['methods'][method]['score_scale']
        probe=raw.detach().clone().requires_grad_(True);grad=torch.autograd.grad(eot.score(probe,method).sum(),probe)[0]
        if not torch.isfinite(grad).all():raise ValueError('nonfinite raw detector gradient')
        gradient_info=dict(finite=True,l1=float(grad.abs().sum()),zero=bool(grad.abs().sum()==0))
        for steps,restarts in (((200,3),(400,5)) if a.trial_index==0 else ((200,3),)):
            for weight in (1.,):
                if done('detector_aware',.2,dict(detector=method,weight=weight,steps=steps,restarts=restarts)):continue
                trace=[];synchronize();start=time.perf_counter()
                delta=defense_aware_pgd(raw,target,eot.logits,(lambda z:z.flatten(1).sum(1)*0) if weight==0 else (lambda z:eot.score(z,method)),threshold,scale,eps,weight,steps,restarts,a.seed,audit=trace.append)
                synchronize();elapsed=time.perf_counter()-start
                settings=dict(detector=method,weight=weight,steps=steps,restarts=restarts,gradient=gradient_info,objective_score_not_computed_when_weight_zero=True)
                traces.append(dict(settings=settings,trace=trace))
                record('detector_aware',.2,delta,settings,elapsed)
    expected_settings=34 if a.trial_index==0 else27
    if len(rows)!=expected_settings:raise ValueError('incomplete or unexpected setting coverage')
    (out/f'attack_audit_complete_seed{a.seed}.json').write_text(json.dumps(dict(**metadata,n_settings=len(rows),status='independent character shard complete'),indent=2))
if __name__=='__main__':main()
