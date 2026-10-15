"""Fit source-only network references on Kaggle; artifacts remain in cloud."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scripts.train_openbmi_source import validate_source_policy
from p3a.data.loaders.openbmi import build_trials
from p3a.preproc.pipeline import Preprocessor
from p3a.data.epochs import gather_epochs,downsample_onsets,epoch_window
from p3a.models.eegnet import EEGNet
from p3a.defenses.network_character import encoder_features
from p3a.defenses.network_reference_scores import source_reference_indices,NetworkMahalanobis


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','source-json','raw-dir','schedule','out-dir','git-commit'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    # Load only our own trusted private source-job checkpoint, never arbitrary user models.
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    c=checkpoint['config'];validate_source_policy(c)
    if checkpoint['seed']!=a.seed or a.seed not in c['seeds']:raise ValueError('source checkpoint seed mismatch')
    provenance=json.loads(Path(a.source_json).read_text())
    if provenance['seed']!=a.seed or provenance['config']!=c:raise ValueError('checkpoint/provenance configuration mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.manual_seed(a.seed);torch.set_num_threads(2)
    pp=c['preprocessing'];pre=Preprocessor(62,pp['raw_fs'],band=tuple(pp['band_hz']),target_fs=pp['target_fs'],car=pp['car'])
    pre.load_state_dict(checkpoint['preprocessor']);pre.to(device)
    model=EEGNet(62,epoch_window(pre.fs_out),**{k:v for k,v in c['decoder'].items() if k!='architecture'}).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    schedule=loadmat(a.schedule,simplify_cells=True)['rc_order'][:5]
    expected={v['subject']:v for v in provenance['source_files']};features=[];labels=[];owners=[];identities=[];files=[]
    if set(expected)!=set(range(1,19)):raise ValueError('exact18 source provenance files required')
    for subject in range(1,19):
        path=Path(a.raw_dir)/f'sess01_subj{subject:02d}_EEG_ERP.mat';sha=digest(path)
        if sha!=expected[subject]['sha256']:raise ValueError('source EEG provenance hash mismatch')
        data=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
        trials=build_trials(data,schedule,'NEURAL_NETWORKS_AND_DEEP_LEARNING',subject,1,'train')
        if [t.trial_id for t in trials]!=expected[subject]['trial_ids']:raise ValueError('source identity mismatch')
        for trial in trials:
            with torch.no_grad():
                signal=pre(torch.as_tensor(trial.eeg,device=device).unsqueeze(0))
                on=downsample_onsets(torch.as_tensor(trial.onsets,device=device).long().unsqueeze(0),pre.q)
                epochs=gather_epochs(signal,on,epoch_window(pre.fs_out))[0]
                features.append(encoder_features(model,epochs))
            labels.extend(trial.target_flash_mask().astype(int).tolist());owners.extend([subject]*60)
            identities.extend([trial.trial_id+f':flash{i:02d}' for i in range(60)])
        files.append(dict(subject=subject,sha256=sha));del data,trials
        print('source references',subject,flush=True)
    x=np.concatenate(features);y=np.asarray(labels);owners=np.asarray(owners)
    indices=source_reference_indices(owners,y,c['reference_statistics']['reference_epoch_limit'],a.seed)
    ref=x[indices];ref_y=y[indices]
    if not np.isfinite(ref).all():raise ValueError('nonfinite source reference features')
    md=NetworkMahalanobis().fit(ref,ref_y)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'references_seed{a.seed}.npz',features=ref,labels=ref_y,means=md.means,precision=md.precision,classes=md.classes)
    info=dict(seed=a.seed,git_commit=a.git_commit,runner_sha256=digest(__file__),checkpoint_sha256=digest(a.checkpoint),schedule_sha256=digest(a.schedule),source_files=files,
        artifact_sha256=digest(out/f'references_seed{a.seed}.npz'),reference_count=len(ref),feature_dimension=ref.shape[1],reference_identities=[identities[i] for i in indices],
        subject_class_counts={f'{s}:{label}':int(np.sum((owners[indices]==s)&(ref_y==label))) for s in range(1,19) for label in (0,1)},
        scope='source-only references, no target outcomes; NPZ remains Kaggle',config=c)
    (out/f'references_seed{a.seed}.json').write_text(json.dumps(info,indent=2))
if __name__=='__main__':main()
