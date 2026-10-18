import pytest
from p3a.eval.shard_checkpoint import load_shard,save_shard,setting_key

def test_atomic_resume_binding_and_duplicates(tmp_path):
    p=tmp_path/'checkpoint.json';meta={'seed':1,'input_sha':'a'}
    assert load_shard(p,meta)==([],[])
    row=dict(attack='PGD',budget=.2,settings={'steps':200,'restarts':3},applied_linf=.1,raw_epsilon=.2,scores={'md':1.})
    save_shard(p,meta,[row],[])
    assert load_shard(p,meta)==([row],[])
    with pytest.raises(ValueError):load_shard(p,dict(meta,seed=2))
    save_shard(p,meta,[row,row],[])
    with pytest.raises(ValueError,match='duplicate'):load_shard(p,meta)
    assert setting_key('PGD',.2,{'steps':200,'restarts':3})!=setting_key('PGD',.1,{'steps':200,'restarts':3})

def test_independent_attack_runner_parses():
    import ast
    from pathlib import Path
    ast.parse((Path(__file__).parents[1]/'scripts/evaluate_openbmi_independent_attacks.py').read_text())
