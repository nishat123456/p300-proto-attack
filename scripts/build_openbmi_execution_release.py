"""Kaggle-only metadata release generation; no EEG or performance inspection."""
import argparse,hashlib,json
from pathlib import Path
from p3a.data.openbmi_manifest import build_subject_manifest
from p3a.data.openbmi_release import execution_code_hashes,load_execution_release

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def find(obj,key):
    if isinstance(obj,dict):
        if key in obj:return obj[key]
        for value in obj.values():
            found=find(value,key)
            if found is not None:return found
    elif isinstance(obj,list):
        for value in obj:
            found=find(value,key)
            if found is not None:return found

def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('release generation must run on Kaggle')
    p=argparse.ArgumentParser()
    for name in ('protocol','coverage-1-51','coverage-52-54','integration-proof','portable-proof','out-dir','git-commit'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();protocol=json.loads(Path(a.protocol).read_text())
    if protocol['status']!='frozen_independent_execution':raise ValueError('final prospectively committed protocol required')
    if protocol['subject_roles']['evaluation']!=list(range(37,55)) or protocol['seeds']!=[20261013,20261014,20261015]:raise ValueError('frozen roles/seeds changed')
    integration=json.loads(Path(a.integration_proof).read_text());portable=json.loads(Path(a.portable_proof).read_text())
    if integration.get('subject')!=19 or integration.get('seeds')!=protocol['seeds']:raise ValueError('actual full validation integration proof required')
    if portable.get('returncode')!=0:raise ValueError('passing cloud portable evidence required')
    records={}
    for path in (a.coverage_1_51,a.coverage_52_54):
        for row in json.loads(Path(path).read_text()):
            key=(row['subject'],row['split'])
            if key in records and records[key]!=row:raise ValueError('conflicting metadata records')
            records[key]=row
    if set(records)!={(s,phase) for s in range(1,55) for phase in ('train','test')}:raise ValueError('complete54-subject metadata required')
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=False)
    pp=out/'protocol.json';pp.write_bytes(Path(a.protocol).read_bytes());ph=digest(pp)
    provenance={name:digest(getattr(a,name)) for name in ('coverage_1_51','coverage_52_54','integration_proof','portable_proof')}
    flow=[]
    for subject in range(1,55):
        train,test=records[(subject,'train')],records[(subject,'test')]
        if train['session']!=1 or test['session']!=1:raise ValueError('session changed')
        if subject==52:
            if find(train,'original_order_mismatch_count')!=528 or find(train,'loader_error')!='schedule and true-character labels disagree':raise ValueError('subject52 missingness evidence changed')
            flow.append(dict(subject=52,status='calibrated estimand unavailable; retained in evaluation flow'));continue
        if find(train,'original_order_mismatch_count')!=0 or find(test,'original_order_mismatch_count')!=0:raise ValueError('metadata label mismatch')
        if train['file_sha256']!=test['file_sha256']:raise ValueError('raw identity mismatch')
        m=build_subject_manifest(subject,find(train,'loader_trial_ids'),find(test,'loader_trial_ids'),train['file_sha256'],ph)
        m['status']='immutable metadata allocation bound to final protocol'
        mp=out/f'manifest_s{subject:02d}.json';mp.write_text(json.dumps(m,indent=2)+'\n')
        if subject>=37:
            receipt=dict(status='frozen_independent_execution',protocol_sha256=ph,manifest_sha256=digest(mp),execution_code_sha256=execution_code_hashes(),evidence_sha256=provenance,git_commit=a.git_commit)
            rp=out/f'receipt_s{subject:02d}.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
            for seed in protocol['seeds']:load_execution_release(pp,mp,rp,subject,seed,train['file_sha256'])
        flow.append(dict(subject=subject,status='metadata eligible',role=m['role']))
    (out/'release_summary.json').write_text(json.dumps(dict(protocol_sha256=ph,git_commit=a.git_commit,evidence_sha256=provenance,execution_code_sha256=execution_code_hashes(),flow=flow,scope='metadata-only release; no performance outcomes'),indent=2)+'\n')
if __name__=='__main__':main()
