import numpy as np
import pytest
from p3a.eval.reliability_metrics import exact_binomial_interval,clean_rejection_accounting,paired_subject_bootstrap

def test_denominators_include_errors_and_rejections():
    r=clean_rejection_accounting(np.array([1,0,1,0],dtype=bool),np.array([0,0,1,1],dtype=bool))
    assert r['observed_far']==.5 and r['accepted_wrong_rate']==.25 and r['accepted_correct_rate']==.25
    assert r['conditional_accepted_accuracy']==.5 and r['iid_retry_overhead']==1

def test_all_rejected_has_undefined_accepted_accuracy():
    r=clean_rejection_accounting(np.array([True]),np.array([True]))
    assert r['conditional_accepted_accuracy'] is None and r['iid_retry_overhead'] is None

def test_intervals_extremes_and_invalid_counts():
    assert exact_binomial_interval(0,36)[0]==0 and exact_binomial_interval(36,36)[1]==1
    with pytest.raises(ValueError):exact_binomial_interval(2,1)

def test_pairing_preserves_constant_effect():
    r=paired_subject_bootstrap([.2,.4,.6],[.1,.3,.5],7)
    assert np.allclose(r['ci95'],[.1,.1]) and np.isclose(r['mean_difference'],.1)
    assert r==paired_subject_bootstrap([.2,.4,.6],[.1,.3,.5],7)
