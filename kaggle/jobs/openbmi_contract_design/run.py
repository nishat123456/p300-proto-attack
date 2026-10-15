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
    config=dict(seed=args.seed, subjects=[52,53,54], sessions=[1],
                selection='fixed subjects52-54 structural completion and identity diagnosis; no performance fitting',
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
    for name,value in json.loads(base64.b64decode('eyJzcmMvcDNhL2RhdGEvbG9hZGVycy9vcGVuYm1pLnB5IjogIlwiXCJcIkJvdW5kZWQgT3BlbkJNSSBjaGFyYWN0ZXIgbG9hZGVyIGNhbmRpZGF0ZTsgc3RvcmVkIGV2ZW50IG9yZGVyIHByZXNlcnZlZC5cIlwiXCJcbmltcG9ydCBudW1weSBhcyBucFxuZnJvbSBwM2EuZGF0YS5sYXlvdXQgaW1wb3J0IExheW91dFxuZnJvbSBwM2EuZGF0YS50cmlhbCBpbXBvcnQgVHJpYWxcblxuTEFZT1VUID0gTGF5b3V0KCdvcGVuYm1pX3NjaGVkdWxlMzYnLCdBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWjEyMzQ1Njc4OV8nLDYsNilcblxuXG5kZWYgYnVpbGRfdHJpYWxzKGRhdGEsIHNjaGVkdWxlLCBwaHJhc2UsIHN1YmplY3QsIHNlc3Npb24sIHNwbGl0KTpcbiAgICBpZiBsZW4oc2NoZWR1bGUpPDU6IHJhaXNlIFZhbHVlRXJyb3IoJ2ZpdmUgc2NoZWR1bGUgZW50cmllcyByZXF1aXJlZCcpXG4gICAgZm9yIGEgaW4gc2NoZWR1bGVbOjVdOlxuICAgICAgICBhPW5wLmFzYXJyYXkoYSlcbiAgICAgICAgaWYgYS5zaGFwZSE9KDEyLDYpIG9yIG5vdCBucC5pc2Zpbml0ZShhKS5hbGwoKSBvciBub3QgbnAuYWxsKGE9PW5wLmZsb29yKGEpKTpcbiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2Zpbml0ZSBpbnRlZ2VyMTJ4NiBzY2hlZHVsZSByZXF1aXJlZCcpXG4gICAgZ3JvdXBzX3Jhdz1ucC5jb25jYXRlbmF0ZShbbnAuYXNhcnJheShhLGR0eXBlPWludCkgZm9yIGEgaW4gc2NoZWR1bGVbOjVdXSxheGlzPTApXG4gICAgaWYgZ3JvdXBzX3Jhdy5zaGFwZSAhPSAoNjAsNikgb3IgZ3JvdXBzX3Jhdy5taW4oKTwxIG9yIGdyb3Vwc19yYXcubWF4KCk+MzY6XG4gICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2ludmFsaWQgZml2ZS1zZXF1ZW5jZSBzY2hlZHVsZScpXG4gICAgZ3JvdXBzPW5wLnplcm9zKCg2MCwzNiksZHR5cGU9Ym9vbClcbiAgICBmb3IgaSxyb3cgaW4gZW51bWVyYXRlKGdyb3Vwc19yYXcpOlxuICAgICAgICBpZiBsZW4oc2V0KHJvdykpIT02OiByYWlzZSBWYWx1ZUVycm9yKCdkdXBsaWNhdGUgc3ltYm9sIHdpdGhpbiBmbGFzaCcpXG4gICAgICAgIGdyb3Vwc1tpLHJvdy0xXT1UcnVlXG4gICAgaWYgbm90IG5wLmFsbChncm91cHMuc3VtKGF4aXM9MCk9PTEwKTogcmFpc2UgVmFsdWVFcnJvcignaW52YWxpZCBzeW1ib2wgY292ZXJhZ2UnKVxuICAgIHg9bnAuYXNhcnJheShkYXRhWyd4J10pOyB0PW5wLmFzYXJyYXkoZGF0YVsndCddKS5yYXZlbCgpXG4gICAgbGFiZWxzPW5wLmFzYXJyYXkoZGF0YVsneV9kZWMnXSkucmF2ZWwoKTsgZnM9ZmxvYXQoZGF0YVsnZnMnXSlcbiAgICBjaGFubmVscz1bc3RyKHYpIGZvciB2IGluIG5wLmFzYXJyYXkoZGF0YVsnY2hhbiddKS5yYXZlbCgpXVxuICAgIGlmIG5vdCBucC5pc2Zpbml0ZShmcykgb3IgZnM8PTA6IHJhaXNlIFZhbHVlRXJyb3IoJ3Bvc2l0aXZlIGZpbml0ZSBzYW1wbGluZyByYXRlIHJlcXVpcmVkJylcbiAgICBpZiBub3QgcGhyYXNlIG9yIGFueShjIG5vdCBpbiBMQVlPVVQuc3ltYm9scyBmb3IgYyBpbiBwaHJhc2UpOiByYWlzZSBWYWx1ZUVycm9yKCdpbnZhbGlkIHBocmFzZScpXG4gICAgaWYgeC5uZGltIT0yIG9yIHguc2hhcGVbMV0hPWxlbihjaGFubmVscykgb3IgbGVuKHNldChjaGFubmVscykpIT1sZW4oY2hhbm5lbHMpOlxuICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCd1bmFsaWduZWQgb3IgZHVwbGljYXRlIGNoYW5uZWxzJylcbiAgICBpZiBub3QgbnAuaXNmaW5pdGUodCkuYWxsKCkgb3Igbm90IG5wLmlzaW4obGFiZWxzLFsxLDJdKS5hbGwoKTpcbiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcignaW52YWxpZCBldmVudCB2YWx1ZXMgb3IgYmluYXJ5IGxhYmVscycpXG4gICAgaWYgbm90IG5wLmFsbCh0PT1ucC5mbG9vcih0KSkgb3IgbGVuKHQpIT02MCpsZW4ocGhyYXNlKSBvciBsZW4obGFiZWxzKSE9bGVuKHQpOlxuICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCd1bmFsaWduZWQgaW50ZWdlciBldmVudC9jaGFyYWN0ZXIgY291bnRzJylcbiAgICB0PXQuYXN0eXBlKG5wLmludDY0KS0xXG4gICAgaWYgeC5uZGltIT0yIG9yIG5vdCBucC5pc2Zpbml0ZSh4KS5hbGwoKSBvciBucC5hbnkodDwwKSBvciBucC5hbnkodD49bGVuKHgpKTpcbiAgICAgICAgcmFpc2UgVmFsdWVFcnJvcignaW52YWxpZCBzaWduYWwvaW5kZXggYm91bmRzJylcbiAgICBpZiBsZW4oc2V0KHQudG9saXN0KCkpKSE9bGVuKHQpOiByYWlzZSBWYWx1ZUVycm9yKCdkdXBsaWNhdGUgZXZlbnQgaW5kaWNlcycpXG4gICAgdHJpYWxzPVtdXG4gICAgZm9yIGksc3ltYm9sIGluIGVudW1lcmF0ZShwaHJhc2UpOlxuICAgICAgICBvbj10W2kqNjA6KGkrMSkqNjBdOyBsYWI9bGFiZWxzW2kqNjA6KGkrMSkqNjBdXG4gICAgICAgIGlmIG5vdCBucC5hcnJheV9lcXVhbChncm91cHNbOixMQVlPVVQuaW5kZXgoc3ltYm9sKV0sbGFiPT0xKTpcbiAgICAgICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ3NjaGVkdWxlIGFuZCB0cnVlLWNoYXJhY3RlciBsYWJlbHMgZGlzYWdyZWUnKVxuICAgICAgICBzdGFydD1tYXgoMCxpbnQob24ubWluKCkpLWludCguMipmcykpOyBlbmQ9aW50KG9uLm1heCgpKStpbnQoZnMpKzFcbiAgICAgICAgaWYgZW5kPmxlbih4KTogcmFpc2UgVmFsdWVFcnJvcigncG9zdC1mbGFzaCBlcG9jaCBleGNlZWRzIHJlY29yZGluZycpXG4gICAgICAgIHRyaWFscy5hcHBlbmQoVHJpYWwoZWVnPXhbc3RhcnQ6ZW5kXS5ULmFzdHlwZShucC5mbG9hdDMyKSxmcz1mcyxcbiAgICAgICAgICAgIG9uc2V0cz1vbi1zdGFydCxmbGFzaF9ncm91cHM9Z3JvdXBzLmNvcHkoKSxmbGFzaF9jb2Rlcz1ucC50aWxlKG5wLmFyYW5nZSgxLDEzKSw1KSxcbiAgICAgICAgICAgIHRhcmdldF9zeW1ib2w9c3ltYm9sLGxheW91dD1MQVlPVVQsc3ViamVjdF9pZD1mJ3N7c3ViamVjdDowMmR9X3Nlc3Npb257c2Vzc2lvbn0nLFxuICAgICAgICAgICAgZGF0YXNldD0nb3BlbmJtaV9jYW5kaWRhdGUnLHRyaWFsX2luZGV4PWksc291cmNlX3NwbGl0PXNwbGl0LFxuICAgICAgICAgICAgY2hhbm5lbF9uYW1lcz1bc3RyKHYpIGZvciB2IGluIG5wLmFzYXJyYXkoZGF0YVsnY2hhbiddKS5yYXZlbCgpXSkpXG4gICAgcmV0dXJuIHRyaWFsc1xuIiwgInRlc3RzL3Rlc3Rfb3BlbmJtaS5weSI6ICJpbXBvcnQgbnVtcHkgYXMgbnBcbmltcG9ydCBweXRlc3RcbmZyb20gcDNhLmRhdGEubG9hZGVycy5vcGVuYm1pIGltcG9ydCBidWlsZF90cmlhbHNcblxuZGVmIGZpeHR1cmUoKTpcbiAgICByb3dzPW5wLmFyYW5nZSgxLDM3KS5yZXNoYXBlKDYsNilcbiAgICBzY2hlZHVsZT1bbnAuY29uY2F0ZW5hdGUoW3Jvd3Mscm93cy5UXSldKjVcbiAgICBtYXNrPW5wLnRpbGUobnAucl9bVHJ1ZSxucC56ZXJvcyg1LGR0eXBlPWJvb2wpLFRydWUsbnAuemVyb3MoNSxkdHlwZT1ib29sKV0sNSlcbiAgICB0PW5wLmFyYW5nZSgyMDEsODAxLDEwKTsgdFswXSx0WzFdPXRbMV0sdFswXVxuICAgIHJldHVybiBkaWN0KHg9bnAuemVyb3MoKDIwMDAsMikpLHQ9dCx5X2RlYz1ucC53aGVyZShtYXNrLDEsMiksZnM9MTAwLGNoYW49WydDeicsJ1B6J10pLHNjaGVkdWxlXG5cbmRlZiB0ZXN0X3ByZXNlcnZlc19vcmRlcl9hbmRfaW5kZXgoKTpcbiAgICBkYXRhLHM9Zml4dHVyZSgpOyB0cmlhbHM9YnVpbGRfdHJpYWxzKGRhdGEscywnQScsMSwxLCd0cmFpbicpXG4gICAgYXNzZXJ0IGxlbih0cmlhbHMpPT0xIGFuZCB0cmlhbHNbMF0ub25zZXRzWzBdPnRyaWFsc1swXS5vbnNldHNbMV1cbiAgICBhc3NlcnQgdHJpYWxzWzBdLm5fZmxhc2hlcz09NjAgYW5kIHRyaWFsc1swXS50YXJnZXRfZmxhc2hfbWFzaygpLnN1bSgpPT0xMFxuXG5AcHl0ZXN0Lm1hcmsucGFyYW1ldHJpemUoJ2tpbmQnLFsnbmVnYXRpdmUnLCdkdXBsaWNhdGUnLCdsYWJlbHMnLCdlbmQnXSlcbmRlZiB0ZXN0X3JlamVjdHNfaW52YWxpZChraW5kKTpcbiAgICBkLHM9Zml4dHVyZSgpXG4gICAgaWYga2luZD09J25lZ2F0aXZlJzogZFsndCddWzBdPTBcbiAgICBpZiBraW5kPT0nZHVwbGljYXRlJzogZFsndCddWzBdPWRbJ3QnXVsxXVxuICAgIGlmIGtpbmQ9PSdsYWJlbHMnOiBkWyd5X2RlYyddWzBdPTEgaWYgZFsneV9kZWMnXVswXT09MiBlbHNlIDJcbiAgICBpZiBraW5kPT0nZW5kJzogZFsneCddPWRbJ3gnXVs6ODAwXVxuICAgIHdpdGggcHl0ZXN0LnJhaXNlcyhWYWx1ZUVycm9yKTogYnVpbGRfdHJpYWxzKGQscywnQScsMSwxLCd0cmFpbicpXG5cbkBweXRlc3QubWFyay5wYXJhbWV0cml6ZSgna2luZCcsWydmc196ZXJvJywnZnNfbmFuJywnbGFiZWwzJywndGltZV9uYW4nLCdmcmFjdGlvbicsJ2NoYW5uZWxzJywnc2NoZWR1bGVfZnJhY3Rpb24nLCdzY2hlZHVsZV9zaG9ydCcsJ3BocmFzZSddKVxuZGVmIHRlc3RfaW5wdXRfY29udHJhY3Qoa2luZCk6XG4gICAgZCxzPWZpeHR1cmUoKVxuICAgIHBocmFzZT0nQSdcbiAgICBpZiBraW5kPT0nZnNfemVybyc6IGRbJ2ZzJ109MFxuICAgIGlmIGtpbmQ9PSdmc19uYW4nOiBkWydmcyddPWZsb2F0KCduYW4nKVxuICAgIGlmIGtpbmQ9PSdsYWJlbDMnOiBkWyd5X2RlYyddWzBdPTNcbiAgICBpZiBraW5kPT0ndGltZV9uYW4nOiBkWyd0J109ZFsndCddLmFzdHlwZShmbG9hdCk7ZFsndCddWzBdPWZsb2F0KCduYW4nKVxuICAgIGlmIGtpbmQ9PSdmcmFjdGlvbic6IGRbJ3QnXT1kWyd0J10uYXN0eXBlKGZsb2F0KTtkWyd0J11bMF0rPS41XG4gICAgaWYga2luZD09J2NoYW5uZWxzJzogZFsnY2hhbiddPVsnQ3onXVxuICAgIGlmIGtpbmQ9PSdzY2hlZHVsZV9mcmFjdGlvbic6IHM9W2EuYXN0eXBlKGZsb2F0KSBmb3IgYSBpbiBzXTtzWzBdWzAsMF09MS41XG4gICAgaWYga2luZD09J3NjaGVkdWxlX3Nob3J0Jzogcz1zWzo0XVxuICAgIGlmIGtpbmQ9PSdwaHJhc2UnOiBwaHJhc2U9Jz8nXG4gICAgd2l0aCBweXRlc3QucmFpc2VzKFZhbHVlRXJyb3IpOiBidWlsZF90cmlhbHMoZCxzLHBocmFzZSwxLDEsJ3RyYWluJylcbiIsICJzcmMvcDNhL2RhdGEvbGF5b3V0LnB5IjogIlwiXCJcIlNwZWxsZXIgbGF5b3V0cyBhbmQgdGhlIGZsYXNoLWdyb3VwIHJlcHJlc2VudGF0aW9uLlxuXG5BIGZsYXNoIGdyb3VwIGlzIHRoZSBzZXQgb2Ygc3ltYm9scyBsaXQgYnkgb25lIGZsYXNoLiBSb3cvY29sdW1uIHNwZWxsZXJzIGFyZVxudGhlIHNwZWNpYWwgY2FzZSB3aGVyZSBlYWNoIGdyb3VwIGlzIG9uZSByb3cgb3Igb25lIGNvbHVtbi5cblwiXCJcIlxuZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9uc1xuXG5mcm9tIGRhdGFjbGFzc2VzIGltcG9ydCBkYXRhY2xhc3NcblxuaW1wb3J0IG51bXB5IGFzIG5wXG5cblxuQGRhdGFjbGFzcyhmcm96ZW49VHJ1ZSlcbmNsYXNzIExheW91dDpcbiAgICBsYXlvdXRfaWQ6IHN0clxuICAgIHN5bWJvbHM6IHN0clxuICAgIG5fcm93czogaW50XG4gICAgbl9jb2xzOiBpbnRcblxuICAgIEBwcm9wZXJ0eVxuICAgIGRlZiBuX3N5bWJvbHMoc2VsZikgLT4gaW50OlxuICAgICAgICByZXR1cm4gbGVuKHNlbGYuc3ltYm9scylcblxuICAgIGRlZiBpbmRleChzZWxmLCBzeW1ib2w6IHN0cikgLT4gaW50OlxuICAgICAgICBpID0gc2VsZi5zeW1ib2xzLmZpbmQoc3ltYm9sKVxuICAgICAgICBpZiBpIDwgMDpcbiAgICAgICAgICAgIHJhaXNlIEtleUVycm9yKGZcInN5bWJvbCB7c3ltYm9sIXJ9IG5vdCBpbiBsYXlvdXQge3NlbGYubGF5b3V0X2lkfVwiKVxuICAgICAgICByZXR1cm4gaVxuXG4gICAgZGVmIHJvd19jb2woc2VsZiwgc3ltYm9sOiBzdHIpIC0+IHR1cGxlW2ludCwgaW50XTpcbiAgICAgICAgcmV0dXJuIGRpdm1vZChzZWxmLmluZGV4KHN5bWJvbCksIHNlbGYubl9jb2xzKVxuXG5cbiMgQkNJIENvbXBldGl0aW9uIElJSSBEYXRhc2V0IElJOiBjb2RlcyAxLTYgPSBjb2x1bW5zIChsZWZ0IHRvIHJpZ2h0KSwgNy0xMiA9IHJvd3MgKHRvcCB0byBib3R0b20pLlxuQ09NUDMgPSBMYXlvdXQoXCJyYzZ4Nl9jb21wM1wiLCBcIkFCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaMTIzNDU2Nzg5X1wiLCA2LCA2KVxuXG5cbmRlZiByY19jb2RlX3RvX2dyb3VwKGxheW91dDogTGF5b3V0LCBjb2RlOiBpbnQpIC0+IG5wLm5kYXJyYXk6XG4gICAgXCJcIlwiQm9vbGVhbiBtYXNrIG92ZXIgc3ltYm9scyBsaXQgYnkgYSByb3cvY29sdW1uIHN0aW11bHVzIGNvZGUgKENvbXAgSUlJIGNvbnZlbnRpb24pLlwiXCJcIlxuICAgIGdyb3VwID0gbnAuemVyb3MobGF5b3V0Lm5fc3ltYm9scywgZHR5cGU9Ym9vbClcbiAgICBpZiAxIDw9IGNvZGUgPD0gbGF5b3V0Lm5fY29sczpcbiAgICAgICAgZ3JvdXBbY29kZSAtIDEgOjogbGF5b3V0Lm5fY29sc10gPSBUcnVlXG4gICAgZWxpZiBsYXlvdXQubl9jb2xzIDwgY29kZSA8PSBsYXlvdXQubl9jb2xzICsgbGF5b3V0Lm5fcm93czpcbiAgICAgICAgciA9IGNvZGUgLSBsYXlvdXQubl9jb2xzIC0gMVxuICAgICAgICBncm91cFtyICogbGF5b3V0Lm5fY29scyA6IChyICsgMSkgKiBsYXlvdXQubl9jb2xzXSA9IFRydWVcbiAgICBlbHNlOlxuICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGZcImludmFsaWQgcm93L2NvbHVtbiBjb2RlIHtjb2RlfVwiKVxuICAgIHJldHVybiBncm91cFxuXG5cbmRlZiBzeW1ib2xfcmNfY29kZXMobGF5b3V0OiBMYXlvdXQsIHN5bWJvbDogc3RyKSAtPiB0dXBsZVtpbnQsIGludF06XG4gICAgXCJcIlwiKHJvd19jb2RlLCBjb2xfY29kZSkgZm9yIGEgc3ltYm9sIHVuZGVyIHRoZSBDb21wIElJSSBjb252ZW50aW9uLlwiXCJcIlxuICAgIHIsIGMgPSBsYXlvdXQucm93X2NvbChzeW1ib2wpXG4gICAgcmV0dXJuIGxheW91dC5uX2NvbHMgKyAxICsgciwgYyArIDFcbiIsICJzcmMvcDNhL2RhdGEvdHJpYWwucHkiOiAiXCJcIlwiVW5pZmllZCB0cmlhbCByZXByZXNlbnRhdGlvbiBzaGFyZWQgYnkgZXZlcnkgZGF0YXNldCBsb2FkZXIuXCJcIlwiXG5mcm9tIF9fZnV0dXJlX18gaW1wb3J0IGFubm90YXRpb25zXG5cbmZyb20gZGF0YWNsYXNzZXMgaW1wb3J0IGRhdGFjbGFzcywgZmllbGRcblxuaW1wb3J0IG51bXB5IGFzIG5wXG5cbmZyb20gLmxheW91dCBpbXBvcnQgTGF5b3V0XG5cblxuQGRhdGFjbGFzc1xuY2xhc3MgVHJpYWw6XG4gICAgXCJcIlwiT25lIGNoYXJhY3Rlci1zZWxlY3Rpb24gdHJpYWwuXG5cbiAgICBlZWc6ICAgICAgICAgIHJhdyBFRUcgW2NoYW5uZWxzLCB0aW1lXSwgZmxvYXQzMiwgYmVmb3JlIGFueSBwcmVwcm9jZXNzaW5nXG4gICAgb25zZXRzOiAgICAgICBmbGFzaCBvbnNldCBzYW1wbGUgaW5kaWNlcyBbRl1cbiAgICBmbGFzaF9ncm91cHM6IGJvb2wgW0YsIG5fc3ltYm9sc10sIHN5bWJvbHMgbGl0IGJ5IGVhY2ggZmxhc2hcbiAgICBmbGFzaF9jb2RlczogIGRhdGFzZXQtbmF0aXZlIHN0aW11bHVzIGNvZGUgcGVyIGZsYXNoIFtGXSAocm93L2NvbCBjb2RlIGZvciBDb21wIElJSSlcbiAgICBcIlwiXCJcblxuICAgIGVlZzogbnAubmRhcnJheVxuICAgIGZzOiBmbG9hdFxuICAgIG9uc2V0czogbnAubmRhcnJheVxuICAgIGZsYXNoX2dyb3VwczogbnAubmRhcnJheVxuICAgIGZsYXNoX2NvZGVzOiBucC5uZGFycmF5XG4gICAgdGFyZ2V0X3N5bWJvbDogc3RyXG4gICAgbGF5b3V0OiBMYXlvdXRcbiAgICBzdWJqZWN0X2lkOiBzdHJcbiAgICBkYXRhc2V0OiBzdHJcbiAgICB0cmlhbF9pbmRleDogaW50XG4gICAgc291cmNlX3NwbGl0OiBzdHIgICMgb2ZmaWNpYWwgc3BsaXQgaW4gdGhlIHNvdXJjZSByZWxlYXNlOiBcInRyYWluXCIgb3IgXCJ0ZXN0XCJcbiAgICBjaGFubmVsX25hbWVzOiBsaXN0W3N0cl0gPSBmaWVsZChkZWZhdWx0X2ZhY3Rvcnk9bGlzdClcblxuICAgIEBwcm9wZXJ0eVxuICAgIGRlZiB0cmlhbF9pZChzZWxmKSAtPiBzdHI6XG4gICAgICAgIHJldHVybiBmXCJ7c2VsZi5kYXRhc2V0fTp7c2VsZi5zdWJqZWN0X2lkfTp7c2VsZi5zb3VyY2Vfc3BsaXR9OntzZWxmLnRyaWFsX2luZGV4OjAzZH1cIlxuXG4gICAgQHByb3BlcnR5XG4gICAgZGVmIG5fZmxhc2hlcyhzZWxmKSAtPiBpbnQ6XG4gICAgICAgIHJldHVybiBsZW4oc2VsZi5vbnNldHMpXG5cbiAgICBkZWYgdGFyZ2V0X2ZsYXNoX21hc2soc2VsZiwgc3ltYm9sOiBzdHIgfCBOb25lID0gTm9uZSkgLT4gbnAubmRhcnJheTpcbiAgICAgICAgXCJcIlwiRmxhc2hlcyB0aGF0IGNvbnRhaW4gYHN5bWJvbGAgKGRlZmF1bHQ6IHRoZSB0cmlhbCdzIG93biB0YXJnZXQpLlwiXCJcIlxuICAgICAgICBrID0gc2VsZi5sYXlvdXQuaW5kZXgoc3ltYm9sIG9yIHNlbGYudGFyZ2V0X3N5bWJvbClcbiAgICAgICAgcmV0dXJuIHNlbGYuZmxhc2hfZ3JvdXBzWzosIGtdXG4ifQ==')).items():
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    os.environ['PYTHONPATH']=str(code/'src')
    sys.path.insert(0,str(code/'src'))
    subprocess.run([sys.executable,'-m','pytest','-q',str(code/'tests/test_openbmi.py')],check=True)
    from p3a.data.loaders.openbmi import build_trials
    records=[]
    for subject,session in ((52,1),(53,1),(54,1)):
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
            # Diagnostic identities compatible with the fixed schedule only.
            # Never replace known labels or infer physical stimulus provenance.
            membership['schedule_compatible_symbols_by_block']=[
                [symbols[k] for k in range(36)
                 if np.array_equal(block==1,np.any(groups==k+1,axis=1))]
                for block in labels.reshape(-1,60)]
            membership['mismatch_event_indices']=np.flatnonzero(expected_labels!=labels).tolist()
            membership['expected_target_counts_by_block']=(expected_labels.reshape(-1,60)==1).sum(axis=1).tolist()
            membership['stored_target_counts_by_block']=(labels.reshape(-1,60)==1).sum(axis=1).tolist()
            try:
                loader_trials=build_trials(data,schedule_rows,phrase,subject,session,split)
                membership['loader_trial_count']=len(loader_trials)
                membership['loader_trial_ids']=[t.trial_id for t in loader_trials]
                del loader_trials
            except ValueError as exc:
                membership['loader_error']=str(exc)
                membership['compatibility_validated']=False
                # Preserve rejection as evidence; never bypass it in model execution.
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
        (out/'openbmi_coverage_audit.json').write_text(json.dumps(records,indent=2))
        path.unlink()  # Raw EEG never becomes an output artifact.
    (out/'README.md').write_text('Metadata-only audit completed. Character-block and stimulus membership reconstruction remains unresolved; no compatibility or performance guarantee. All calculations executed on Kaggle. Raw EEG removed from working storage after inspection. BNCI2015-003 was not accessed.\n')


if __name__ == '__main__':
    main()
