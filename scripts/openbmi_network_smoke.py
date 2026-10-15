"""Development-only real-data execution check, not benchmark performance."""
import argparse,json,hashlib,tempfile
from pathlib import Path
import numpy as np
import requests
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from p3a.models.eegnet import EEGNet
from p3a.defenses.network_reference_scores import dropout_uncertainty,local_intrinsic_dimension,NetworkMahalanobis
from p3a.defenses.network_character import dropout_probability_draws,encoder_features,mahalanobis_input_adjustment,predicted_character_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20261014);p.add_argument('--out-dir',required=True);a=p.parse_args()
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(2)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    url='https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/session1/s1/sess01_subj01_EEG_ERP.mat'
    with tempfile.TemporaryDirectory(dir='/kaggle/working') as tmp:
        path=Path(tmp)/'source.mat';digest=hashlib.sha256()
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with path.open('wb') as stream:
                for block in response.iter_content(1024*1024):
                    if block: stream.write(block);digest.update(block)
        d=loadmat(path,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
        # First ten development characters train; next three reference; next one query.
        # Provided epochs already baseline-adjusted: diagnostic route, not final raw-attack pipeline.
        from p3a.preproc.pipeline import Preprocessor
        from p3a.data.epochs import downsample_onsets,gather_epochs,epoch_window
        times=np.asarray(d['t']).ravel()[:840].astype(np.int64)-1
        continuous=np.asarray(d['x']);trials=[];onsets=[]
        for block in times.reshape(14,60):
            begin=max(0,int(block.min())-1000);end=int(block.max())+1001
            if end>len(continuous):raise ValueError('raw trial boundary exceeds recording')
            trials.append(continuous[begin:end].T.astype(np.float32));onsets.append(block-begin)
        pre=Preprocessor(62,1000.,target_fs=60.)
        pre.fit_scaling(trials[:10])
        extracted=[]
        for raw,onset in zip(trials,onsets):
            with torch.no_grad():
                signal=pre(torch.as_tensor(raw).unsqueeze(0))
                ds=torch.as_tensor(downsample_onsets(onset,pre.q)).long().unsqueeze(0)
                extracted.append(gather_epochs(signal,ds,epoch_window(pre.fs_out))[0].numpy())
        x=np.concatenate(extracted);y=(np.asarray(d['y_dec']).ravel()[:840]==1).astype(np.float32)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model=EEGNet(x.shape[1],x.shape[2]).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
        for epoch in range(2):
            model.train()
            for ids in np.array_split(np.arange(600),10):
                tx=torch.as_tensor(x[ids],device=device);ty=torch.as_tensor(y[ids],device=device)
                opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(tx),ty);loss.backward();opt.step()
        ref=torch.as_tensor(x[600:780],device=device);query=torch.as_tensor(x[780:840],device=device)
        features=encoder_features(model,ref); qfeatures=encoder_features(model,query)
        md=NetworkMahalanobis().fit(features,y[600:780])
        adjusted=mahalanobis_input_adjustment(model,query,md,.001)
        scores=dict(bu=dropout_uncertainty(dropout_probability_draws(model,query,8,a.seed)),
                    lid=local_intrinsic_dimension(qfeatures,features,20),md=md.score(qfeatures),
                    md_adjusted=md.score(encoder_features(model,adjusted)))
        schedule_url='https://raw.githubusercontent.com/PatternRecognition/OpenBMI/master/GigaScience/function_ERP/random_cell_order.mat'
        response=requests.get(schedule_url,timeout=60);response.raise_for_status();sp=Path(tmp)/'schedule.mat';sp.write_bytes(response.content)
        schedule=loadmat(sp,simplify_cells=True)['rc_order'];rows=np.concatenate([np.asarray(z,dtype=int) for z in schedule[:5]])
        groups=np.zeros((60,36),dtype=bool)
        for i,row in enumerate(rows):groups[i,row-1]=True
        model.eval()
        with torch.no_grad(): evidence=model(query).cpu().numpy()
        checks={name:dict(n_flash_scores=len(v),finite=bool(np.isfinite(v).all()),
            character_score_finite=bool(np.isfinite(predicted_character_score(evidence,v,groups)['score']))) for name,v in scores.items()}
        if not all(c['finite'] and c['character_score_finite'] for c in checks.values()):raise RuntimeError('nonfinite detector integration')
        result=dict(seed=a.seed,source_commit='4bc41bb plus smoke overlay',subject=1,session=1,phase='train only',
            source_sha256=digest.hexdigest(),device=str(device),preprocessing=pre.to_config(),decoder_epochs=2,fit_characters=list(range(10)),
            reference_characters=list(range(10,13)),query_character=13,checks=checks,
            status='execution smoke only; no accuracy/FAR/TPR/AUC or independent evaluation',
            limitations='raw Preprocessor and gather_epochs execution; no attacks or frozen benchmark;8draws/k20/epsilon.001 are smoke only')
        (out/'network_smoke.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
