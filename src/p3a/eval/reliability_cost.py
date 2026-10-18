"""Measured computation and recorded-marker proxies, never clinical throughput."""
import time
import numpy as np

def marker_duration_seconds(onsets,fs,epoch_seconds=.8):
    events=np.asarray(onsets,dtype=float)
    if events.ndim!=1 or len(events)<2 or not np.isfinite(events).all() or not np.isfinite(fs) or fs<=0 or epoch_seconds<=0:
        raise ValueError('finite event sequence and positive sampling/epoch duration required')
    # Preserve stored order, including documented timestamp reversals.
    duration=float((events[-1]-events[0])/fs+epoch_seconds)
    if duration<=0:raise ValueError('nonpositive recorded first-to-last duration')
    return duration

def measure_inference(callback,synchronize,warmup=5,repeats=30):
    if warmup<0 or repeats<2:raise ValueError('nonnegative warmup and at least2 repeats required')
    for _ in range(warmup):callback();synchronize()
    elapsed=[]
    for _ in range(repeats):
        synchronize();start=time.perf_counter();callback();synchronize()
        elapsed.append(time.perf_counter()-start)
    return dict(warmup=warmup,repeats=repeats,seconds=elapsed,median_seconds=float(np.median(elapsed)),p95_seconds=float(np.quantile(elapsed,.95)),scope='batch1 computation only; not physical latency or clinical spelling throughput')
