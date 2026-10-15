"""Independent clean-character calibration, no attack labels or score tuning."""
import numpy as np
from p3a.data.openbmi_partitions import validate_partition

METHODS=('confidence','entropy','bu','lid','md','md_adjusted','kde')


def fit_clean_calibration(scores,score_ids,partition,bu_draws=100):
    validate_partition(partition)
    if list(score_ids)!=list(partition['calibration']):
        raise ValueError('exact ordered independent20 calibration IDs required')
    if bu_draws!=100:raise ValueError('calibration and reported BU scores require100 draws')
    if set(scores)!=set(METHODS):raise ValueError('all7 declared methods required')
    result={'calibration_ids':list(score_ids),'evaluation_ids':list(partition['evaluation']),
            'bu_draws':100,'reject_rule':'strictly greater, ties retained','methods':{}}
    for method in METHODS:
        values=np.asarray(scores[method],dtype=float)
        if values.shape!=(20,) or not np.isfinite(values).all():
            raise ValueError('20 finite clean character scores per method required')
        median=float(np.median(values));mad=float(np.median(np.abs(values-median)))
        result['methods'][method]={'thresholds':{str(alpha):float(np.quantile(values,1-alpha,method='higher'))
            for alpha in (.01,.05,.1)},'median':median,'score_scale':max(mad,1e-6),'mad_floor':1e-6}
    return result


def reject_evaluation_scores(calibration,method,alpha,scores,score_ids,bu_draws=100):
    if method not in METHODS or str(alpha) not in calibration['methods'][method]['thresholds']:
        raise ValueError('undeclared detector or operating point')
    ids=list(score_ids)
    if len(set(ids))!=len(ids) or not set(ids).issubset(calibration['evaluation_ids']):
        raise ValueError('unique evaluation IDs required, never calibration reuse')
    if method=='bu' and bu_draws!=calibration['bu_draws']:
        raise ValueError('BU scoring distribution differs from clean calibration')
    values=np.asarray(scores,dtype=float)
    if values.shape!=(len(ids),) or not np.isfinite(values).all():raise ValueError('aligned finite scores required')
    return values>calibration['methods'][method]['thresholds'][str(alpha)]
