import pytest
from p3a.eval.reliability_cost import marker_duration_seconds,measure_inference

def test_marker_proxy_preserves_stored_endpoints():
    assert marker_duration_seconds([100,90,300],1000)==pytest.approx(1.)
    with pytest.raises(ValueError):marker_duration_seconds([100,200],0)
    with pytest.raises(ValueError):marker_duration_seconds([float('nan'),200],1000)

def test_timing_calls_and_scope():
    calls=[];sync=[]
    r=measure_inference(lambda:calls.append(1),lambda:sync.append(1),warmup=2,repeats=3)
    assert len(calls)==5 and len(sync)==8
    assert len(r['seconds'])==3 and r['median_seconds']>=0 and r['p95_seconds']>=0
    assert 'not physical' in r['scope']
