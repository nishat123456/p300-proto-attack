"""Deterministic one-subject/one-seed cloud execution with exact input binding."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.fit_openbmi_references import digest
from p3a.data.openbmi_release import load_execution_release


def main():
    parser=argparse.ArgumentParser()
    for name in ('protocol','manifest','release-receipt','checkpoint','references','reference-json','template','raw-file','schedule','out-dir','git-commit'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--subject',type=int,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--validation-integration',action='store_true');a=parser.parse_args()
    manifest,ph,mh=load_execution_release(a.protocol,a.manifest,a.release_receipt,a.subject,a.seed,digest(a.raw_file),validation_integration=a.validation_integration)
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    binding=dict(seed=a.seed,subject=a.subject,protocol_sha256=ph,manifest_sha256=mh,git_commit=a.git_commit,
        runner_sha256=digest(__file__),inputs={n:digest(getattr(a,n.replace('-','_'))) for n in ('checkpoint','references','reference-json','template','raw-file','schedule')})
    receipt=out/'subject_execution.json'
    if receipt.exists():
        state=json.loads(receipt.read_text())
        if state['binding']!=binding:raise ValueError('subject resume input/protocol/code mismatch')
    else:state=dict(binding=binding,completed={})
    def persist():
        temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(receipt)
    persist()
    scripts=Path(__file__).parent
    common=[]
    for name in ('protocol','manifest','release-receipt','checkpoint','references','raw-file','schedule','out-dir','git-commit','subject','seed'):
        common.extend(['--'+name,str(getattr(a,name.replace('-','_')))])
    if a.validation_integration:common.append('--validation-integration')
    cal=out/f'calibration_s{a.subject:02d}_seed{a.seed}.json'
    pre=out/f'user_preprocessor_s{a.subject:02d}_seed{a.seed}.pt'
    clean=out/f'validation_clean_s{a.subject:02d}_seed{a.seed}.json'
    def stage(name,command,artifacts):
        if name in state['completed']:
            if state['completed'][name]!={str(p.relative_to(out)):digest(p) for p in artifacts}:raise ValueError('completed stage output changed: '+name)
            return
        subprocess.run([sys.executable,str(scripts/command[0])]+common+command[1:],check=True)
        state['completed'][name]={str(p.relative_to(out)):digest(p) for p in artifacts};persist()
    stage('calibration',['calibrate_openbmi_independent.py','--reference-json',a.reference_json],[cal,pre])
    bound=['--calibration',str(cal),'--user-preprocessor',str(pre)]
    stage('clean',['evaluate_openbmi_independent_clean.py']+bound,[clean])
    # Every declared identity is executed, regardless of clean correctness or attack outcome.
    for index,trial_id in enumerate(manifest['partition']['evaluation']):
        folder=out/f's{a.subject:02d}_seed{a.seed}_trial{index:03d}'
        proof=folder/f'attack_audit_complete_seed{a.seed}.json'
        rows=folder/f'attack_audit_seed{a.seed}.json'
        stage('attack:'+trial_id,['evaluate_openbmi_independent_attacks.py']+bound+['--reference-json',a.reference_json,'--template',a.template,'--trial-index',str(index)],[proof,rows])
    if len(state['completed'])!=38:raise ValueError('incomplete subject stages')
    state['status']='all36 original clean/attack characters executed; no outcome-selected omissions';persist()
if __name__=='__main__':main()
