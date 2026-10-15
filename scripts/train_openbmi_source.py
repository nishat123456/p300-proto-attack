"""Stage-A source-only decoder trainer. Execute on Kaggle, never locally."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from p3a.data.loaders.openbmi import build_trials
from p3a.preproc.pipeline import Preprocessor
from p3a.data.epochs import gather_epochs,downsample_onsets,epoch_window
from p3a.models.eegnet import EEGNet


def validate_source_policy(config):
    roles=[config[k] for k in ('development_subjects','validation_subjects','sealed_evaluation_subjects')]
    if [sorted(r) for r in roles]!=[list(range(1,19)),list(range(19,37)),list(range(37,55))]:
        raise ValueError('stage-A exact independent subject roles required')
    if len(set(config['seeds']))!=3:raise ValueError('three distinct source seeds required')
    if config['source_training']['official_phase']!='train':raise ValueError('official train only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--raw-dir',required=True)
    p.add_argument('--schedule',required=True);p.add_argument('--git-commit',required=True);p.add_argument('--out-dir',required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());validate_source_policy(c)
    if a.seed not in c['seeds']:raise ValueError('seed not preregistered in source config')
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);torch.set_num_threads(2)
    raw=Path(a.raw_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    schedule=loadmat(a.schedule,simplify_cells=True)['rc_order'][:5]
    phrase='NEURAL_NETWORKS_AND_DEEP_LEARNING';trials=[];provenance=[]
    for subject in c['development_subjects']:
        path=raw/f'sess01_subj{subject:02d}_EEG_ERP.mat'
        # Explicitly request only source official training phase, no test labels.
        d=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
        group=build_trials(d,schedule,phrase,subject,1,'train')
        if len(group)!=33:raise ValueError('unexpected source character count')
        trials.extend(group)
        with path.open('rb') as handle:
            file_hash=hashlib.file_digest(handle,'sha256').hexdigest()
        provenance.append(dict(subject=subject,file=path.name,sha256=file_hash,size=path.stat().st_size,trial_ids=[t.trial_id for t in group]))
        del d
    pp=c['preprocessing'];pre=Preprocessor(62,pp['raw_fs'],band=tuple(pp['band_hz']),target_fs=pp['target_fs'],car=pp['car'])
    pre.fit_scaling([t.eeg for t in trials]);xs=[];ys=[];owners=[]
    for t in trials:
        with torch.no_grad():
            signal=pre(torch.as_tensor(t.eeg).unsqueeze(0));on=torch.as_tensor(downsample_onsets(t.onsets,pre.q)).long().unsqueeze(0)
            xs.append(gather_epochs(signal,on,epoch_window(pre.fs_out))[0].numpy())
        ys.append(t.target_flash_mask().astype(np.float32));owners.extend([t.subject_id]*60)
    x=np.concatenate(xs);y=np.concatenate(ys);del trials,xs,ys
    if not np.isfinite(x).all() or not 0<y.mean()<1:raise ValueError('invalid source epochs')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args={k:v for k,v in c['decoder'].items() if k!='architecture'}
    model=EEGNet(x.shape[1],x.shape[2],**args).to(device);opt=torch.optim.Adam(model.parameters(),lr=c['source_training']['learning_rate'])
    pos_weight=torch.tensor(float((y==0).sum()/(y==1).sum()),device=device)
    losses=[];batch=c['source_training']['flash_batch_size']
    for epoch in range(c['source_training']['epochs']):
        model.train();order=rng.permutation(len(x));total=0.
        for start in range(0,len(order),batch):
            idx=order[start:start+batch];tx=torch.as_tensor(x[idx],device=device);ty=torch.as_tensor(y[idx],device=device)
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx),ty,pos_weight=pos_weight)
            loss.backward();opt.step();total+=float(loss.detach())*len(idx)
        losses.append(total/len(x));print('source epoch',epoch+1,losses[-1],flush=True)
    torch.save(dict(model=model.cpu().state_dict(),preprocessor=pre.state_dict(),config=c,seed=a.seed),out/f'source_seed{a.seed}.pt')
    (out/f'source_seed{a.seed}.json').write_text(json.dumps(dict(seed=a.seed,git_commit=a.git_commit,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),schedule_sha256=hashlib.sha256(Path(a.schedule).read_bytes()).hexdigest(),config=c,preprocessor=pre.to_config(),source_files=provenance,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training_losses=losses,n_flash_epochs=len(x),
        positive_weight=float(pos_weight.cpu()),status='source training only, no target evaluation'),indent=2))
if __name__=='__main__':main()
