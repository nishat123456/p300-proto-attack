import json,sys,copy
from pathlib import Path
import pytest
from scripts.aggregate_openbmi_reliability import main
from p3a.defenses.clean_calibration import METHODS
from scripts.fit_openbmi_references import digest

def fixture(root):
    protocol=root/'protocol.json';protocol.write_text(json.dumps({'seeds':[20261013,20261014,20261015]}));ph=digest(protocol)
    for seed in (20261013,20261014,20261015):
        folder=root/f'seed{seed}';folder.mkdir();summaries=[]
        for m in METHODS:
            for alpha in (.01,.05,.1):summaries.append(dict(subject=19,seed=seed,method=m,nominal_far=alpha,n_characters=36,observed_far=0.,clean_accuracy=1.,accepted_correct_rate=1.,accepted_wrong_rate=0.))
        p=folder/f'validation_clean_s19_seed{seed}.json';p.write_text(json.dumps(dict(subject=19,seed=seed,protocol_sha256=ph,manifest_sha256='manifest',summaries=summaries)))
        p.with_suffix('.csv').write_text('method,nominal_far,anomaly\n'+''.join(f'{m},0.05,0\n' for m in METHODS for _ in range(36)))
        for index in range(36):
            target=index%2==0;rows=[]
            for budget in (.05,.1,.2,.3,.5):
                for attack in ('FGSM','Gaussian_control','source_template','PGD'):
                    rows.append(dict(attack=attack,budget=budget,settings={},applied_linf=.1,raw_epsilon=.2,targeted_success=target,rejected_at_nominal5={m:False for m in METHODS},scores={m:1. for m in METHODS}))
            for m in METHODS:
                for steps in ((200,400) if index==0 else (200,)):
                    rows.append(dict(attack='detector_aware',budget=.2,settings={'detector':m,'steps':steps,'weight':1.,'restarts':3 if steps==200 else 5},applied_linf=.1,raw_epsilon=.2,targeted_success=target,rejected_at_nominal5={m:False for m in METHODS},scores={m:1. for m in METHODS}))
            folder2=folder/str(index);folder2.mkdir()
            (folder2/f'attack_audit_seed{seed}.json').write_text(json.dumps(dict(subject=19,seed=seed,trial_id=f'openbmi_candidate:s19_session1:test:{index:03d}',protocol_sha256=ph,manifest_sha256='manifest',rows=rows)))
    return protocol

def invoke(root,protocol,monkeypatch):
    monkeypatch.setattr(sys,'argv',['aggregate','--root',str(root),'--protocol',str(protocol),'--out-dir',str(root/'output'),'--git-commit','synthetic','--seed','20261013','--validation-integration'])
    main()

def test_all_attempt_denominators_and_missing_shard(tmp_path,monkeypatch):
    protocol=fixture(tmp_path);invoke(tmp_path,protocol,monkeypatch)
    result=json.loads((tmp_path/'output/reliability_aggregate.json').read_text())
    assert len(result['per_seed'])==3*147
    assert all(r['n_attempts']==36 and r['n_targeted']==18 and r['defended_targeted_success_rate']==.5 and r['conditional_tpr']==0 for r in result['per_seed'])
    assert result['paired_descriptive_comparisons']==[]
    assert all(r['detector_unaware_auc_all_attempts']==1 for r in result['per_seed'] if r['attack']!='detector_aware')
    (tmp_path/'seed20261013/35/attack_audit_seed20261013.json').unlink()
    with pytest.raises(ValueError,match='36'):invoke(tmp_path,protocol,monkeypatch)

def test_execution_scripts_parse():
    import ast
    for name in ('run_openbmi_independent_subject.py','aggregate_openbmi_reliability.py','benchmark_openbmi_source_cost.py'):
        ast.parse((Path(__file__).parents[1]/'scripts'/name).read_text())
