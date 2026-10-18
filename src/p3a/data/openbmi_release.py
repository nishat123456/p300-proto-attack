"""Fail-closed independent execution binding; candidate artifacts never unlock."""
import hashlib,json
from pathlib import Path
from p3a.data.openbmi_manifest import build_subject_manifest,validate_evaluation_binding

def load_execution_release(protocol_path,manifest_path,receipt_path,subject,seed,raw_sha256):
    protocol_bytes=Path(protocol_path).read_bytes();manifest_bytes=Path(manifest_path).read_bytes()
    protocol=json.loads(protocol_bytes);manifest=json.loads(manifest_bytes);receipt=json.loads(Path(receipt_path).read_text())
    ph=hashlib.sha256(protocol_bytes).hexdigest();mh=hashlib.sha256(manifest_bytes).hexdigest()
    if receipt.get('status')!='frozen_independent_execution' or protocol.get('status')!='frozen_independent_execution':
        raise ValueError('candidate protocol or absent final release; independent gate CLOSED')
    if receipt.get('protocol_sha256')!=ph or receipt.get('manifest_sha256')!=mh:
        raise ValueError('release content hash mismatch')
    if seed not in protocol['seeds'] or seed not in (20261013,20261014,20261015):
        raise ValueError('undeclared model seed')
    validate_evaluation_binding(manifest,ph,subject,manifest['partition']['evaluation'])
    if manifest['raw_sha256']!=raw_sha256:raise ValueError('raw file identity mismatch')
    partition=manifest['partition']
    train=sorted(partition['normalization']+partition['calibration'])
    expected=build_subject_manifest(subject,train,partition['evaluation'],raw_sha256,ph,20261013)
    for key in ('partition','targets','seed','session'):
        if manifest[key]!=expected[key]:raise ValueError('manifest identity/target inconsistency: '+key)
    if protocol['subject_roles']['evaluation']!=list(range(37,55)) or subject==52:
        raise ValueError('fixed evaluation flow or metadata eligibility changed')
    return manifest,ph,mh
