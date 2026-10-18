"""Dedicated frozen-manifest independent runner. No candidate release accepted."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scripts.train_openbmi_source import validate_source_policy
from scripts.fit_openbmi_references import digest
from p3a.data.loaders.openbmi import build_trials
from p3a.data.openbmi_partitions import candidate_character_partition
from p3a.data.openbmi_release import load_execution_release
from p3a.data.epochs import epoch_window
from p3a.models.eegnet import EEGNet
from p3a.preproc.pipeline import Preprocessor
from p3a.defenses.raw_detector_pipeline import RawDetectorPipeline
from p3a.defenses.clean_calibration import METHODS,fit_clean_calibration


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','references','reference-json','raw-file','schedule','out-dir','git-commit'):
        p.add_argument('--'+name,required=True)
    for name in ('protocol','manifest','release-receipt'):p.add_argument('--'+name,required=True)
    p.add_argument('--validation-integration',action='store_true',help='fixed subject19 only; never opens sealed evaluation')
    p.add_argument('--subject',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    torch.manual_seed(a.seed);torch.set_num_threads(2)
    manifest,protocol_hash,manifest_hash=load_execution_release(a.protocol,a.manifest,a.release_receipt,a.subject,a.seed,digest(a.raw_file),validation_integration=a.validation_integration)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    c=checkpoint['config'];validate_source_policy(c)
    if checkpoint['seed']!=a.seed or a.seed not in c['seeds']:raise ValueError('source model seed mismatch')
    provenance=json.loads(Path(a.reference_json).read_text())
    if provenance['seed']!=a.seed or provenance['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('reference checkpoint mismatch')
    if provenance['artifact_sha256']!=digest(a.references) or provenance['schedule_sha256']!=digest(a.schedule):
        raise ValueError('reference or schedule provenance mismatch')
    # Load only official train. Test IDs identify a proposed disjoint role, not loaded data.
    data=loadmat(a.raw_file,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
    schedule=loadmat(a.schedule,simplify_cells=True)['rc_order'][:5]
    trials=build_trials(data,schedule,'NEURAL_NETWORKS_AND_DEEP_LEARNING',a.subject,1,'train')
    train_ids=[t.trial_id for t in trials]
    test_ids=[f'openbmi_candidate:s{a.subject:02d}_session1:test:{i:03d}' for i in range(36)]
    partition=manifest['partition']
    if sorted(train_ids)!=sorted(partition['normalization']+partition['calibration']):raise ValueError('train identities differ from frozen manifest')
    by_id={t.trial_id:t for t in trials}
    norm=[by_id[i] for i in partition['normalization']]
    pp=c['preprocessing'];pre=Preprocessor(62,pp['raw_fs'],band=tuple(pp['band_hz']),target_fs=pp['target_fs'],car=pp['car'])
    pre.fit_scaling([t.eeg for t in norm])
    scale=float(np.median(np.concatenate([t.eeg for t in norm],axis=1).std(axis=1,ddof=0)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid normalization-only raw budget scale')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');pre.to(device)
    model=EEGNet(62,epoch_window(pre.fs_out),**{k:v for k,v in c['decoder'].items() if k!='architecture'}).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    with np.load(a.references,allow_pickle=False) as refs:
        features=refs['features'].copy();labels=refs['labels'].copy();means=refs['means'].copy();precision=refs['precision'].copy()
    scores={m:[] for m in METHODS}
    for identity in partition['calibration']:
        trial=by_id[identity]
        pipeline=RawDetectorPipeline(pre,model,torch.as_tensor(trial.onsets,device=device).long().unsqueeze(0),
            torch.as_tensor(trial.flash_groups,device=device).bool().unsqueeze(0),features,means,precision,20,100,a.seed,.001,
            reference_labels=labels,kde_bandwidth=.5).to(device)
        raw=torch.as_tensor(trial.eeg,device=device).unsqueeze(0)
        for method in METHODS:
            value=float(pipeline.score(raw,method).detach().cpu().item())
            scores[method].append(value)
        print('clean calibration character',identity,flush=True)
    result=fit_clean_calibration(scores,partition['calibration'],partition,100)
    result.update(seed=a.seed,subject=a.subject,git_commit=a.git_commit,runner_sha256=digest(__file__),
        checkpoint_sha256=digest(a.checkpoint),reference_sha256=digest(a.references),raw_sha256=digest(a.raw_file),
        schedule_sha256=digest(a.schedule),normalization_ids=partition['normalization'],normalization_raw_scale=scale,
        calibration_scores=scores,protocol_sha256=protocol_hash,manifest_sha256=manifest_hash,scope='independent official-train-only calibration under frozen release',
        partition_status='exact frozen manifest')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(preprocessor=pre.cpu().state_dict(),config=pre.to_config(),seed=a.seed,subject=a.subject,
        normalization_ids=partition['normalization']),out/f'user_preprocessor_s{a.subject:02d}_seed{a.seed}.pt')
    (out/f'calibration_s{a.subject:02d}_seed{a.seed}.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
