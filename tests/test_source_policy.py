import copy
import pytest
from scripts.train_openbmi_source import validate_source_policy

def config():
    return dict(development_subjects=list(range(1,19)),validation_subjects=list(range(19,37)),sealed_evaluation_subjects=list(range(37,55)),seeds=[1,2,3],source_training={'official_phase':'train'})

def test_exact_source_roles(): validate_source_policy(config())

@pytest.mark.parametrize('kind',['sealed','overlap','test_phase','seed'])
def test_rejects_leakage(kind):
    c=config()
    if kind=='sealed':c['development_subjects'][0]=37
    if kind=='overlap':c['validation_subjects'][0]=1
    if kind=='test_phase':c['source_training']['official_phase']='test'
    if kind=='seed':c['seeds']=[1,1,2]
    with pytest.raises(ValueError):validate_source_policy(c)
