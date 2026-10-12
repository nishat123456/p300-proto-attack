"""Cloud-only OpenBMI structure and clean-FAR feasibility audit."""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path


def upper_zero(n, confidence=0.95):
    if n < 1:
        raise ValueError("positive independent evaluation count required")
    return 1 - (1 - confidence) ** (1 / n)


def event_audit(times, labels, samples):
    import numpy as np
    times, labels = np.asarray(times).ravel(), np.asarray(labels).ravel()
    return dict(n_events=len(times), aligned=len(times) == len(labels),
                increasing=bool(np.all(np.diff(times) > 0)),
                within_recording=bool(len(times) and times.min() >= 0 and times.max() <= samples),
                label_counts={str(v): int((labels == v).sum()) for v in np.unique(labels)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    out = Path('/kaggle/working/out'); out.mkdir(exist_ok=True)
    raw = Path('/kaggle/working/raw_openbmi'); raw.mkdir(exist_ok=True)
    # Independent sample design arithmetic only; not fitted EEG performance.
    design = [dict(n=n, zero_alarm_upper_95=upper_zero(n), rank_resolution=1/(n+1))
              for n in (5, 6, 20, 33, 36, 59, 69, 100, 200, 299, 500)]
    (out/'sample_size_design.json').write_text(json.dumps(dict(
        assumptions='IID Bernoulli evaluation; exchangeable rank calibration. Neither guaranteed for EEG.',
        confidence=0.95, nominal_far=0.05, rows=design,
        zero_alarm_min_n_95_for_5_percent=math.ceil(math.log(0.05)/math.log(0.95)),
        warning='Group pooling and flash repetitions do not imply per-user independent characters.'), indent=2))
    assert abs(upper_zero(1) - .95) < 1e-12
    assert upper_zero(59) <= .05 < upper_zero(58)
    assert event_audit([1,2], [1,2], 3)['aligned']
    assert not event_audit([2,1], [1,2], 3)['increasing']
    assert not event_audit([1], [1,2], 3)['aligned']
    (out/'focused_tests.json').write_text(json.dumps({'checks_passed':5,
        'scope':'five focused arithmetic/schema assertions; metadata-only follow-up, not full repository suite'}))
    import numpy as np
    import requests
    from scipy.io import loadmat, whosmat
    base='https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/'
    config=dict(seed=args.seed, subjects=[2,3], sessions=[1],
                selection='predeclared subjects2/3 session1; no outcome-based selection',
                source_commit='c335257', runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python=platform.python_version(),
                stage='event ordering and identity metadata only; no performance evaluation',
                reference='MOABB Lee2019_ERP documentation; nasion reference documented, not measured',
                no_raw_output=True, frozen_splits_untouched=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    schedule_url='https://raw.githubusercontent.com/PatternRecognition/OpenBMI/master/GigaScience/function_ERP/random_cell_order.mat'
    response=requests.get(schedule_url,timeout=60); response.raise_for_status()
    schedule_path=raw/'schedule.mat'; schedule_path.write_bytes(response.content)
    schedule_mat=loadmat(schedule_path,simplify_cells=True)
    keys=[k for k in schedule_mat if not k.startswith('__')]
    schedule_value=schedule_mat[keys[0]] if len(keys)==1 else None
    schedule_rows=[]
    if schedule_value is not None:
        cells=np.asarray(schedule_value,dtype=object)
        for item in cells.ravel():
            a=np.asarray(item)
            if a.ndim==2: schedule_rows.append(a.astype(int))
    schedule_available_shapes=[list(a.shape) for a in schedule_rows]
    schedule_rows=schedule_rows[:5]  # Original ERP_performance uses nSeq1..Nsequence=5.
    schedule_ok=bool(len(schedule_rows)==5 and all(a.shape==(12,6) and a.min()>=1 and a.max()<=36 for a in schedule_rows))
    if schedule_ok:
        schedule_ok=all(np.all(np.bincount(a.ravel(),minlength=37)[1:]==2) for a in schedule_rows)
    config['schedule']=dict(url=schedule_url,sha256=hashlib.sha256(response.content).hexdigest(),keys=keys,available_shapes=schedule_available_shapes,selection='first five entries as original decoder Nsequence=5',valid_structure=schedule_ok,shapes=[list(a.shape) for a in schedule_rows])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    import base64,os
    code=Path('/kaggle/working/loader_code');code.mkdir(exist_ok=True)
    for name,value in json.loads(base64.b64decode('eyJzcmMvcDNhL2RhdGEvbG9hZGVycy9vcGVuYm1pLnB5IjogIlwiXCJcIkJvdW5kZWQgT3BlbkJNSSBjaGFyYWN0ZXIgbG9hZGVyIGNhbmRpZGF0ZTsgc3RvcmVkIGV2ZW50IG9yZGVyIHByZXNlcnZlZC5cIlwiXCJcbmltcG9ydCBudW1weSBhcyBucFxuZnJvbSBwM2EuZGF0YS5sYXlvdXQgaW1wb3J0IExheW91dFxuZnJvbSBwM2EuZGF0YS50cmlhbCBpbXBvcnQgVHJpYWxcblxuTEFZT1VUID0gTGF5b3V0KCdvcGVuYm1pX3NjaGVkdWxlMzYnLCdBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWjEyMzQ1Njc4OV8nLDYsNilcblxuXG5kZWYgYnVpbGRfdHJpYWxzKGRhdGEsIHNjaGVkdWxlLCBwaHJhc2UsIHN1YmplY3QsIHNlc3Npb24sIHNwbGl0KTpcbiAgICBncm91cHNfcmF3PW5wLmNvbmNhdGVuYXRlKFtucC5hc2FycmF5KGEsZHR5cGU9aW50KSBmb3IgYSBpbiBzY2hlZHVsZVs6NV1dLGF4aXM9MClcbiAgICBpZiBncm91cHNfcmF3LnNoYXBlICE9ICg2MCw2KSBvciBncm91cHNfcmF3Lm1pbigpPDEgb3IgZ3JvdXBzX3Jhdy5tYXgoKT4zNjpcbiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcignaW52YWxpZCBmaXZlLXNlcXVlbmNlIHNjaGVkdWxlJylcbiAgICBncm91cHM9bnAuemVyb3MoKDYwLDM2KSxkdHlwZT1ib29sKVxuICAgIGZvciBpLHJvdyBpbiBlbnVtZXJhdGUoZ3JvdXBzX3Jhdyk6XG4gICAgICAgIGlmIGxlbihzZXQocm93KSkhPTY6IHJhaXNlIFZhbHVlRXJyb3IoJ2R1cGxpY2F0ZSBzeW1ib2wgd2l0aGluIGZsYXNoJylcbiAgICAgICAgZ3JvdXBzW2kscm93LTFdPVRydWVcbiAgICBpZiBub3QgbnAuYWxsKGdyb3Vwcy5zdW0oYXhpcz0wKT09MTApOiByYWlzZSBWYWx1ZUVycm9yKCdpbnZhbGlkIHN5bWJvbCBjb3ZlcmFnZScpXG4gICAgeD1ucC5hc2FycmF5KGRhdGFbJ3gnXSk7IHQ9bnAuYXNhcnJheShkYXRhWyd0J10pLnJhdmVsKClcbiAgICBsYWJlbHM9bnAuYXNhcnJheShkYXRhWyd5X2RlYyddKS5yYXZlbCgpOyBmcz1mbG9hdChkYXRhWydmcyddKVxuICAgIGlmIG5vdCBucC5hbGwodD09bnAuZmxvb3IodCkpIG9yIGxlbih0KSE9NjAqbGVuKHBocmFzZSkgb3IgbGVuKGxhYmVscykhPWxlbih0KTpcbiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcigndW5hbGlnbmVkIGludGVnZXIgZXZlbnQvY2hhcmFjdGVyIGNvdW50cycpXG4gICAgdD10LmFzdHlwZShucC5pbnQ2NCktMVxuICAgIGlmIHgubmRpbSE9MiBvciBub3QgbnAuaXNmaW5pdGUoeCkuYWxsKCkgb3IgbnAuYW55KHQ8MCkgb3IgbnAuYW55KHQ+PWxlbih4KSk6XG4gICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2ludmFsaWQgc2lnbmFsL2luZGV4IGJvdW5kcycpXG4gICAgaWYgbGVuKHNldCh0LnRvbGlzdCgpKSkhPWxlbih0KTogcmFpc2UgVmFsdWVFcnJvcignZHVwbGljYXRlIGV2ZW50IGluZGljZXMnKVxuICAgIHRyaWFscz1bXVxuICAgIGZvciBpLHN5bWJvbCBpbiBlbnVtZXJhdGUocGhyYXNlKTpcbiAgICAgICAgb249dFtpKjYwOihpKzEpKjYwXTsgbGFiPWxhYmVsc1tpKjYwOihpKzEpKjYwXVxuICAgICAgICBpZiBub3QgbnAuYXJyYXlfZXF1YWwoZ3JvdXBzWzosTEFZT1VULmluZGV4KHN5bWJvbCldLGxhYj09MSk6XG4gICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCdzY2hlZHVsZSBhbmQgdHJ1ZS1jaGFyYWN0ZXIgbGFiZWxzIGRpc2FncmVlJylcbiAgICAgICAgc3RhcnQ9bWF4KDAsaW50KG9uLm1pbigpKS1pbnQoLjIqZnMpKTsgZW5kPWludChvbi5tYXgoKSkraW50KGZzKSsxXG4gICAgICAgIGlmIGVuZD5sZW4oeCk6IHJhaXNlIFZhbHVlRXJyb3IoJ3Bvc3QtZmxhc2ggZXBvY2ggZXhjZWVkcyByZWNvcmRpbmcnKVxuICAgICAgICB0cmlhbHMuYXBwZW5kKFRyaWFsKGVlZz14W3N0YXJ0OmVuZF0uVC5hc3R5cGUobnAuZmxvYXQzMiksZnM9ZnMsXG4gICAgICAgICAgICBvbnNldHM9b24tc3RhcnQsZmxhc2hfZ3JvdXBzPWdyb3Vwcy5jb3B5KCksZmxhc2hfY29kZXM9bnAudGlsZShucC5hcmFuZ2UoMSwxMyksNSksXG4gICAgICAgICAgICB0YXJnZXRfc3ltYm9sPXN5bWJvbCxsYXlvdXQ9TEFZT1VULHN1YmplY3RfaWQ9Zidze3N1YmplY3Q6MDJkfV9zZXNzaW9ue3Nlc3Npb259JyxcbiAgICAgICAgICAgIGRhdGFzZXQ9J29wZW5ibWlfY2FuZGlkYXRlJyx0cmlhbF9pbmRleD1pLHNvdXJjZV9zcGxpdD1zcGxpdCxcbiAgICAgICAgICAgIGNoYW5uZWxfbmFtZXM9W3N0cih2KSBmb3IgdiBpbiBucC5hc2FycmF5KGRhdGFbJ2NoYW4nXSkucmF2ZWwoKV0pKVxuICAgIHJldHVybiB0cmlhbHNcbiIsICJ0ZXN0cy90ZXN0X29wZW5ibWkucHkiOiAiaW1wb3J0IG51bXB5IGFzIG5wXG5pbXBvcnQgcHl0ZXN0XG5mcm9tIHAzYS5kYXRhLmxvYWRlcnMub3BlbmJtaSBpbXBvcnQgYnVpbGRfdHJpYWxzXG5cbmRlZiBmaXh0dXJlKCk6XG4gICAgcm93cz1ucC5hcmFuZ2UoMSwzNykucmVzaGFwZSg2LDYpXG4gICAgc2NoZWR1bGU9W25wLmNvbmNhdGVuYXRlKFtyb3dzLHJvd3MuVF0pXSo1XG4gICAgbWFzaz1ucC50aWxlKG5wLnJfW1RydWUsbnAuemVyb3MoNSxkdHlwZT1ib29sKSxUcnVlLG5wLnplcm9zKDUsZHR5cGU9Ym9vbCldLDUpXG4gICAgdD1ucC5hcmFuZ2UoMjAxLDgwMSwxMCk7IHRbMF0sdFsxXT10WzFdLHRbMF1cbiAgICByZXR1cm4gZGljdCh4PW5wLnplcm9zKCgyMDAwLDIpKSx0PXQseV9kZWM9bnAud2hlcmUobWFzaywxLDIpLGZzPTEwMCxjaGFuPVsnQ3onLCdQeiddKSxzY2hlZHVsZVxuXG5kZWYgdGVzdF9wcmVzZXJ2ZXNfb3JkZXJfYW5kX2luZGV4KCk6XG4gICAgZGF0YSxzPWZpeHR1cmUoKTsgdHJpYWxzPWJ1aWxkX3RyaWFscyhkYXRhLHMsJ0EnLDEsMSwndHJhaW4nKVxuICAgIGFzc2VydCBsZW4odHJpYWxzKT09MSBhbmQgdHJpYWxzWzBdLm9uc2V0c1swXT50cmlhbHNbMF0ub25zZXRzWzFdXG4gICAgYXNzZXJ0IHRyaWFsc1swXS5uX2ZsYXNoZXM9PTYwIGFuZCB0cmlhbHNbMF0udGFyZ2V0X2ZsYXNoX21hc2soKS5zdW0oKT09MTBcblxuQHB5dGVzdC5tYXJrLnBhcmFtZXRyaXplKCdraW5kJyxbJ25lZ2F0aXZlJywnZHVwbGljYXRlJywnbGFiZWxzJywnZW5kJ10pXG5kZWYgdGVzdF9yZWplY3RzX2ludmFsaWQoa2luZCk6XG4gICAgZCxzPWZpeHR1cmUoKVxuICAgIGlmIGtpbmQ9PSduZWdhdGl2ZSc6IGRbJ3QnXVswXT0wXG4gICAgaWYga2luZD09J2R1cGxpY2F0ZSc6IGRbJ3QnXVswXT1kWyd0J11bMV1cbiAgICBpZiBraW5kPT0nbGFiZWxzJzogZFsneV9kZWMnXVswXT0xIGlmIGRbJ3lfZGVjJ11bMF09PTIgZWxzZSAyXG4gICAgaWYga2luZD09J2VuZCc6IGRbJ3gnXT1kWyd4J11bOjgwMF1cbiAgICB3aXRoIHB5dGVzdC5yYWlzZXMoVmFsdWVFcnJvcik6IGJ1aWxkX3RyaWFscyhkLHMsJ0EnLDEsMSwndHJhaW4nKVxuIiwgInNyYy9wM2EvZGF0YS9sYXlvdXQucHkiOiAiXCJcIlwiU3BlbGxlciBsYXlvdXRzIGFuZCB0aGUgZmxhc2gtZ3JvdXAgcmVwcmVzZW50YXRpb24uXG5cbkEgZmxhc2ggZ3JvdXAgaXMgdGhlIHNldCBvZiBzeW1ib2xzIGxpdCBieSBvbmUgZmxhc2guIFJvdy9jb2x1bW4gc3BlbGxlcnMgYXJlXG50aGUgc3BlY2lhbCBjYXNlIHdoZXJlIGVhY2ggZ3JvdXAgaXMgb25lIHJvdyBvciBvbmUgY29sdW1uLlxuXCJcIlwiXG5mcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zXG5cbmZyb20gZGF0YWNsYXNzZXMgaW1wb3J0IGRhdGFjbGFzc1xuXG5pbXBvcnQgbnVtcHkgYXMgbnBcblxuXG5AZGF0YWNsYXNzKGZyb3plbj1UcnVlKVxuY2xhc3MgTGF5b3V0OlxuICAgIGxheW91dF9pZDogc3RyXG4gICAgc3ltYm9sczogc3RyXG4gICAgbl9yb3dzOiBpbnRcbiAgICBuX2NvbHM6IGludFxuXG4gICAgQHByb3BlcnR5XG4gICAgZGVmIG5fc3ltYm9scyhzZWxmKSAtPiBpbnQ6XG4gICAgICAgIHJldHVybiBsZW4oc2VsZi5zeW1ib2xzKVxuXG4gICAgZGVmIGluZGV4KHNlbGYsIHN5bWJvbDogc3RyKSAtPiBpbnQ6XG4gICAgICAgIGkgPSBzZWxmLnN5bWJvbHMuZmluZChzeW1ib2wpXG4gICAgICAgIGlmIGkgPCAwOlxuICAgICAgICAgICAgcmFpc2UgS2V5RXJyb3IoZlwic3ltYm9sIHtzeW1ib2whcn0gbm90IGluIGxheW91dCB7c2VsZi5sYXlvdXRfaWR9XCIpXG4gICAgICAgIHJldHVybiBpXG5cbiAgICBkZWYgcm93X2NvbChzZWxmLCBzeW1ib2w6IHN0cikgLT4gdHVwbGVbaW50LCBpbnRdOlxuICAgICAgICByZXR1cm4gZGl2bW9kKHNlbGYuaW5kZXgoc3ltYm9sKSwgc2VsZi5uX2NvbHMpXG5cblxuIyBCQ0kgQ29tcGV0aXRpb24gSUlJIERhdGFzZXQgSUk6IGNvZGVzIDEtNiA9IGNvbHVtbnMgKGxlZnQgdG8gcmlnaHQpLCA3LTEyID0gcm93cyAodG9wIHRvIGJvdHRvbSkuXG5DT01QMyA9IExheW91dChcInJjNng2X2NvbXAzXCIsIFwiQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY3ODlfXCIsIDYsIDYpXG5cblxuZGVmIHJjX2NvZGVfdG9fZ3JvdXAobGF5b3V0OiBMYXlvdXQsIGNvZGU6IGludCkgLT4gbnAubmRhcnJheTpcbiAgICBcIlwiXCJCb29sZWFuIG1hc2sgb3ZlciBzeW1ib2xzIGxpdCBieSBhIHJvdy9jb2x1bW4gc3RpbXVsdXMgY29kZSAoQ29tcCBJSUkgY29udmVudGlvbikuXCJcIlwiXG4gICAgZ3JvdXAgPSBucC56ZXJvcyhsYXlvdXQubl9zeW1ib2xzLCBkdHlwZT1ib29sKVxuICAgIGlmIDEgPD0gY29kZSA8PSBsYXlvdXQubl9jb2xzOlxuICAgICAgICBncm91cFtjb2RlIC0gMSA6OiBsYXlvdXQubl9jb2xzXSA9IFRydWVcbiAgICBlbGlmIGxheW91dC5uX2NvbHMgPCBjb2RlIDw9IGxheW91dC5uX2NvbHMgKyBsYXlvdXQubl9yb3dzOlxuICAgICAgICByID0gY29kZSAtIGxheW91dC5uX2NvbHMgLSAxXG4gICAgICAgIGdyb3VwW3IgKiBsYXlvdXQubl9jb2xzIDogKHIgKyAxKSAqIGxheW91dC5uX2NvbHNdID0gVHJ1ZVxuICAgIGVsc2U6XG4gICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoZlwiaW52YWxpZCByb3cvY29sdW1uIGNvZGUge2NvZGV9XCIpXG4gICAgcmV0dXJuIGdyb3VwXG5cblxuZGVmIHN5bWJvbF9yY19jb2RlcyhsYXlvdXQ6IExheW91dCwgc3ltYm9sOiBzdHIpIC0+IHR1cGxlW2ludCwgaW50XTpcbiAgICBcIlwiXCIocm93X2NvZGUsIGNvbF9jb2RlKSBmb3IgYSBzeW1ib2wgdW5kZXIgdGhlIENvbXAgSUlJIGNvbnZlbnRpb24uXCJcIlwiXG4gICAgciwgYyA9IGxheW91dC5yb3dfY29sKHN5bWJvbClcbiAgICByZXR1cm4gbGF5b3V0Lm5fY29scyArIDEgKyByLCBjICsgMVxuIiwgInNyYy9wM2EvZGF0YS90cmlhbC5weSI6ICJcIlwiXCJVbmlmaWVkIHRyaWFsIHJlcHJlc2VudGF0aW9uIHNoYXJlZCBieSBldmVyeSBkYXRhc2V0IGxvYWRlci5cIlwiXCJcbmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnNcblxuZnJvbSBkYXRhY2xhc3NlcyBpbXBvcnQgZGF0YWNsYXNzLCBmaWVsZFxuXG5pbXBvcnQgbnVtcHkgYXMgbnBcblxuZnJvbSAubGF5b3V0IGltcG9ydCBMYXlvdXRcblxuXG5AZGF0YWNsYXNzXG5jbGFzcyBUcmlhbDpcbiAgICBcIlwiXCJPbmUgY2hhcmFjdGVyLXNlbGVjdGlvbiB0cmlhbC5cblxuICAgIGVlZzogICAgICAgICAgcmF3IEVFRyBbY2hhbm5lbHMsIHRpbWVdLCBmbG9hdDMyLCBiZWZvcmUgYW55IHByZXByb2Nlc3NpbmdcbiAgICBvbnNldHM6ICAgICAgIGZsYXNoIG9uc2V0IHNhbXBsZSBpbmRpY2VzIFtGXVxuICAgIGZsYXNoX2dyb3VwczogYm9vbCBbRiwgbl9zeW1ib2xzXSwgc3ltYm9scyBsaXQgYnkgZWFjaCBmbGFzaFxuICAgIGZsYXNoX2NvZGVzOiAgZGF0YXNldC1uYXRpdmUgc3RpbXVsdXMgY29kZSBwZXIgZmxhc2ggW0ZdIChyb3cvY29sIGNvZGUgZm9yIENvbXAgSUlJKVxuICAgIFwiXCJcIlxuXG4gICAgZWVnOiBucC5uZGFycmF5XG4gICAgZnM6IGZsb2F0XG4gICAgb25zZXRzOiBucC5uZGFycmF5XG4gICAgZmxhc2hfZ3JvdXBzOiBucC5uZGFycmF5XG4gICAgZmxhc2hfY29kZXM6IG5wLm5kYXJyYXlcbiAgICB0YXJnZXRfc3ltYm9sOiBzdHJcbiAgICBsYXlvdXQ6IExheW91dFxuICAgIHN1YmplY3RfaWQ6IHN0clxuICAgIGRhdGFzZXQ6IHN0clxuICAgIHRyaWFsX2luZGV4OiBpbnRcbiAgICBzb3VyY2Vfc3BsaXQ6IHN0ciAgIyBvZmZpY2lhbCBzcGxpdCBpbiB0aGUgc291cmNlIHJlbGVhc2U6IFwidHJhaW5cIiBvciBcInRlc3RcIlxuICAgIGNoYW5uZWxfbmFtZXM6IGxpc3Rbc3RyXSA9IGZpZWxkKGRlZmF1bHRfZmFjdG9yeT1saXN0KVxuXG4gICAgQHByb3BlcnR5XG4gICAgZGVmIHRyaWFsX2lkKHNlbGYpIC0+IHN0cjpcbiAgICAgICAgcmV0dXJuIGZcIntzZWxmLmRhdGFzZXR9OntzZWxmLnN1YmplY3RfaWR9OntzZWxmLnNvdXJjZV9zcGxpdH06e3NlbGYudHJpYWxfaW5kZXg6MDNkfVwiXG5cbiAgICBAcHJvcGVydHlcbiAgICBkZWYgbl9mbGFzaGVzKHNlbGYpIC0+IGludDpcbiAgICAgICAgcmV0dXJuIGxlbihzZWxmLm9uc2V0cylcblxuICAgIGRlZiB0YXJnZXRfZmxhc2hfbWFzayhzZWxmLCBzeW1ib2w6IHN0ciB8IE5vbmUgPSBOb25lKSAtPiBucC5uZGFycmF5OlxuICAgICAgICBcIlwiXCJGbGFzaGVzIHRoYXQgY29udGFpbiBgc3ltYm9sYCAoZGVmYXVsdDogdGhlIHRyaWFsJ3Mgb3duIHRhcmdldCkuXCJcIlwiXG4gICAgICAgIGsgPSBzZWxmLmxheW91dC5pbmRleChzeW1ib2wgb3Igc2VsZi50YXJnZXRfc3ltYm9sKVxuICAgICAgICByZXR1cm4gc2VsZi5mbGFzaF9ncm91cHNbOiwga11cbiJ9')).items():
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    os.environ['PYTHONPATH']=str(code/'src')
    sys.path.insert(0,str(code/'src'))
    subprocess.run([sys.executable,'-m','pytest','-q',str(code/'tests/test_openbmi.py')],check=True)
    from p3a.data.loaders.openbmi import build_trials
    records=[]
    for subject,session in ((2,1),(3,1)):
        name=f'sess{session:02d}_subj{subject:02d}_EEG_ERP.mat'
        url=base+f'session{session}/s{subject}/'+name
        path=raw/name
        print('Downloading cloud-only metadata source',name,flush=True)
        digest=hashlib.sha256()
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with path.open('wb') as stream:
                for block in response.iter_content(1024*1024):
                    if block: stream.write(block); digest.update(block)
        schema=whosmat(path)
        mat=loadmat(path,simplify_cells=True,
                    variable_names=['EEG_ERP_train','EEG_ERP_test'])
        for split in ('train','test'):
            data=mat[f'EEG_ERP_{split}']
            fields={k:dict(shape=list(np.shape(v)),dtype=str(np.asarray(v).dtype)) for k,v in data.items()}
            x=np.asarray(data['x']); times=np.asarray(data['t']).ravel()
            labels=np.asarray(data['y_dec']).ravel()
            events=event_audit(times,labels,x.shape[0])

            order=np.argsort(times,kind='stable')
            chronological=times[order]
            chronological_labels=labels[order]
            unique_times,counts=np.unique(times,return_counts=True)
            blocks=chronological_labels.reshape(-1,60) if len(times)%60==0 else None
            ordering=dict(duplicate_onsets=int(np.sum(counts-1)),
                nonincreasing_steps=int(np.sum(np.diff(times)<=0)),
                label_transition_count=int(np.sum(np.diff(labels)!=0)),
                original_label_runs=[dict(label=int(labels[i]),start=int(i)) for i in
                    np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1]],
                sorted_gap_quantiles=np.quantile(np.diff(chronological),[0,.5,.9,.99,1]).tolist(),
                strict_zero_based_bounds=bool(times.min()>=0 and times.max()<x.shape[0]),
                strict_matlab_one_based_bounds=bool(times.min()>=1 and times.max()<=x.shape[0]),
                inferred60_block_target_counts=[] if blocks is None else (blocks==1).sum(axis=1).tolist(),
                sorted_order_status='diagnostic only, not validated character or symbol identity',
                label_vocab={k:np.unique(np.asarray(data[k]).astype(str)).tolist()[:40]
                    for k in ('y_dec','y_class')},
                class_mapping=str(data['class']),
                gap_interpretation='large gaps may locate decisions, cannot assign flashed symbols')
            symbols='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_'
            phrase='NEURAL_NETWORKS_AND_DEEP_LEARNING' if split=='train' else 'PATTERN_RECOGNITION_MACHINE_LEARNING'
            membership=dict(schedule_valid_structure=schedule_ok,phrase_length=len(phrase),status='unverified')
            if schedule_ok and len(labels)==len(phrase)*60:
                groups=np.concatenate(schedule_rows,axis=0)
                expected_labels=np.concatenate([np.where(np.any(groups==symbols.index(c)+1,axis=1),1,2) for c in phrase])
                membership.update(original_order_mismatch_count=int(np.sum(expected_labels!=labels)),
                    chronological_order_mismatch_count=int(np.sum(expected_labels!=chronological_labels)),
                    original60_block_monotonic_counts=[int(np.sum(np.diff(b)<=0)) for b in times.reshape(-1,60)],
                    status='label consistency only; does not independently establish timestamp origin or physical schedule execution')

            # Fixed diagnostic samples, never selected by decoder outcome.
            reversals=np.flatnonzero(np.diff(times)<=0)
            reversal_details=[dict(position=int(i),position_in60=int(i%60),
                before=int(times[i]),after=int(times[i+1]),difference=int(times[i+1]-times[i])) for i in reversals]
            epoch=np.asarray(data['smt'])
            ival=np.asarray(data['ival']).ravel()
            selected=sorted(set([0,1,59,60,len(times)-1]+[int(i) for i in reversals[:3]]))
            matches=[]
            for shift in (-2,-1,0,1,2):
                squared=0.; reference=0.; count=0; correlations=[]; centered_error=0.; centered_reference=0.
                for event in selected:
                    indices=times[event]+ival+shift
                    if np.all(indices>=0) and np.all(indices<x.shape[0]) and len(indices)==epoch.shape[0]:
                        a=x[indices.astype(int),:]; b=epoch[:,event,:]
                        ac=a-a.mean(axis=0,keepdims=True); bc=b-b.mean(axis=0,keepdims=True)
                        centered_error+=float(np.sum((ac-bc)**2)); centered_reference+=float(np.sum(bc*bc))
                        squared+=float(np.sum((a-b)**2)); reference+=float(np.sum(b*b)); count+=a.size
                        correlations.append(float(np.corrcoef(a.ravel(),b.ravel())[0,1]))
                matches.append(dict(shift=shift,points=count,centered_relative_l2=None if not centered_reference else float(np.sqrt(centered_error/centered_reference)),rmse=None if not count else float(np.sqrt(squared/count)),
                    relative_l2=None if not reference else float(np.sqrt(squared/reference)),
                    mean_correlation=None if not correlations else float(np.mean(correlations))))
            alignment=dict(reversals=reversal_details,selected_event_indices=selected,
                ival_min=float(ival.min()),ival_max=float(ival.max()),ival_length=len(ival),
                epoch_shape=list(epoch.shape),offset_comparisons=matches,
                formula='x[t + ival + shift]; MATLAB integer indexing predicts shift=-1 if ival are integer sample offsets',
                caveat='No shift automatically chosen. Filtered/baseline-adjusted epochs may differ; correlations do not prove exact alignment.')
            loader_trials=build_trials(data,schedule_rows,phrase,subject,session,split)
            membership['loader_trial_count']=len(loader_trials)
            membership['loader_trial_ids']=[t.trial_id for t in loader_trials]
            del loader_trials
            expected=33 if split=='train' else 36
            records.append(dict(subject=subject,session=session,split=split,url=url,
                file_sha256=digest.hexdigest(),file_bytes=path.stat().st_size,
                top_level_schema=schema,fields=fields,fs=float(np.asarray(data['fs']).squeeze()),
                channel_names=[str(v) for v in np.asarray(data['chan']).ravel()],
                signal_shape=list(x.shape),events=events,ordering=ordering,schedule_membership=membership,alignment=alignment,
                documented_characters=expected,
                blocks_of_60_if_divisible=len(times)//60 if len(times)%60==0 else None,
                character_count_status='inferred block count only; independent decision IDs not yet validated',
                membership_status='unresolved; y_dec Target/NonTarget alone does not define flashed symbols',
                possible_membership_fields=[k for k in data if k not in ('x','t','fs','chan','y_dec','EMG','EMG_index')]))
            print('Metadata inspected',session,split,len(times),flush=True)
        del mat
        (out/'openbmi_loader_audit.json').write_text(json.dumps(records,indent=2))
        path.unlink()  # Raw EEG never becomes an output artifact.
    (out/'README.md').write_text('Metadata-only audit completed. Character-block and stimulus membership reconstruction remains unresolved; no compatibility or performance guarantee. All calculations executed on Kaggle. Raw EEG removed from working storage after inspection. BNCI2015-003 was not accessed.\n')


if __name__ == '__main__':
    main()
