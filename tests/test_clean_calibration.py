import numpy as np
import pytest
from p3a.defenses.clean_calibration import METHODS,fit_clean_calibration,reject_evaluation_scores


def inputs():
    partition={'normalization':[f'n{i}' for i in range(13)],'calibration':[f'c{i}' for i in range(20)],'evaluation':[f'e{i}' for i in range(36)]}
    return {m:np.arange(20,dtype=float) for m in METHODS},partition


def test_quantile_discreteness_ties_and_evaluation_guard():
    scores,partition=inputs();c=fit_clean_calibration(scores,partition['calibration'],partition)
    assert c['methods']['md']['thresholds']['0.01']==19
    assert c['methods']['md']['thresholds']['0.05']==19
    assert c['methods']['md']['thresholds']['0.1']==18
    assert reject_evaluation_scores(c,'md',.05,[19,20],['e0','e1']).tolist()==[False,True]
    with pytest.raises(ValueError):reject_evaluation_scores(c,'md',.05,[20],['c0'])
    with pytest.raises(ValueError):reject_evaluation_scores(c,'bu',.05,[20],['e0'],bu_draws=30)


def test_constant_scale_floor_and_invalid_calibration():
    scores,partition=inputs();scores={m:np.ones(20) for m in METHODS}
    c=fit_clean_calibration(scores,partition['calibration'],partition)
    assert all(v['score_scale']==1e-6 for v in c['methods'].values())
    with pytest.raises(ValueError):fit_clean_calibration(scores,partition['calibration'][::-1],partition)
    with pytest.raises(ValueError):fit_clean_calibration(scores,partition['calibration'],partition,30)
    scores['md'][0]=np.nan
    with pytest.raises(ValueError):fit_clean_calibration(scores,partition['calibration'],partition)


def test_role_overlap_is_rejected_before_score_fit():
    scores,partition=inputs();partition['normalization'][0]='c0'
    with pytest.raises(ValueError):fit_clean_calibration(scores,partition['calibration'],partition)
