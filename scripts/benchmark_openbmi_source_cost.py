"""Source1 fixed training query computation timing, no outcome selection."""
import argparse,json,platform
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from scripts.fit_openbmi_references import digest
from scripts.train_openbmi_source import validate_source_policy
from p3a.data.loaders.openbmi import build_trials
from p3a.data.epochs import epoch_window
from p3a.models.eegnet import EEGNet
from p3a.preproc.pipeline import Preprocessor
from p3a.defenses.raw_detector_pipeline import RawDetectorPipeline
from p3a.defenses.clean_calibration import METHODS
from p3a.eval.reliability_cost import measure_inference,marker_duration_seconds

def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','source-json','references','reference-json','raw-file','schedule','out-dir','git-commit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seed',required=True,type=int);a=parser.parse_args()
    if not torch.cuda.is_available():raise ValueError('prospective GPU timing requires GPU, never substitute CPU silently')
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=checkpoint['config'];validate_source_policy(c)
    source=json.loads(Path(a.source_json).read_text());proof=json.loads(Path(a.reference_json).read_text())
    if checkpoint['seed']!=a.seed or source['seed']!=a.seed or source['config']!=c:raise ValueError('source seed/config mismatch')
    if proof['checkpoint_sha256']!=digest(a.checkpoint) or proof['artifact_sha256']!=digest(a.references) or proof['schedule_sha256']!=digest(a.schedule):raise ValueError('reference provenance mismatch')
    expected=[x for x in source['source_files'] if x['subject']==1]
    if len(expected)!=1 or expected[0]['sha256']!=digest(a.raw_file):raise ValueError('fixed source1 raw identity required')
    schedule=loadmat(a.schedule,simplify_cells=True)['rc_order'][:5]
    data=loadmat(a.raw_file,simplify_cells=True,variable_names=['EEG_ERP_train'])['EEG_ERP_train']
    trials=build_trials(data,schedule,'NEURAL_NETWORKS_AND_DEEP_LEARNING',1,1,'train')
    if [t.trial_id for t in trials]!=expected[0]['trial_ids']:raise ValueError('fixed source identities mismatch')
    trial=trials[0];device=torch.device('cuda');pp=c['preprocessing']
    pre=Preprocessor(62,pp['raw_fs'],band=tuple(pp['band_hz']),target_fs=pp['target_fs'],car=pp['car']).to(device);pre.load_state_dict(checkpoint['preprocessor'])
    model=EEGNet(62,epoch_window(pre.fs_out),**{k:v for k,v in c['decoder'].items() if k!='architecture'}).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    with np.load(a.references,allow_pickle=False) as ref:
        pipeline=RawDetectorPipeline(pre,model,torch.as_tensor(trial.onsets,device=device).long().unsqueeze(0),torch.as_tensor(trial.flash_groups,device=device).bool().unsqueeze(0),ref['features'].copy(),ref['means'].copy(),ref['precision'].copy(),20,100,a.seed,.001,reference_labels=ref['labels'].copy(),kde_bandwidth=.5).to(device)
    raw=torch.as_tensor(trial.eeg,device=device).unsqueeze(0)
    def score(method):
        # Detach each result; adjusted-MD intentionally needs input gradients internally.
        return pipeline.score(raw,method).detach()
    timing={method:measure_inference(lambda m=method:score(m),torch.cuda.synchronize) for method in METHODS}
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    result=dict(seed=a.seed,git_commit=a.git_commit,runner_sha256=digest(__file__),checkpoint_sha256=digest(a.checkpoint),reference_sha256=digest(a.references),raw_sha256=digest(a.raw_file),schedule_sha256=digest(a.schedule),trial_id=trial.trial_id,methods=timing,marker_duration_seconds=marker_duration_seconds(trial.onsets,trial.fs),hardware=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,python_version=platform.python_version(),scope='fixed source1 first train query batch1 synchronized computation; not clinical or physical latency, no performance outcome')
    (out/f'source_cost_seed{a.seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__':main()
