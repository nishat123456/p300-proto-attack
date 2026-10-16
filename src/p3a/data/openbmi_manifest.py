"""Build candidate manifests from cloud-verified loader identities, never EEG."""
import hashlib
from p3a.data.openbmi_partitions import candidate_character_partition,candidate_subject_role

TRAIN_PHRASE='NEURAL_NETWORKS_AND_DEEP_LEARNING'
TEST_PHRASE='PATTERN_RECOGNITION_MACHINE_LEARNING'

def build_subject_manifest(subject,train_ids,test_ids,raw_sha256,protocol_sha256,seed=20261013):
    role=candidate_subject_role(subject)
    for digest in (raw_sha256,protocol_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('explicit lowercase SHA256 provenance required')
    if subject==52:
        raise ValueError('subject52 calibrated manifest unavailable; retain in flow separately')
    expected=lambda phase,n:[f'openbmi_candidate:s{subject:02d}_session1:{phase}:{i:03d}' for i in range(n)]
    if train_ids!=expected('train',33) or test_ids!=expected('test',36):
        raise ValueError('exact cloud-validated stored-order official phase IDs required')
    partition=candidate_character_partition(train_ids,test_ids,seed)
    # The loader layout is authoritative; phrases here only identify intended symbols.
    from p3a.data.loaders.openbmi import LAYOUT
    targets=[]
    for trial_id,symbol in zip(test_ids,TEST_PHRASE):
        intended=LAYOUT.index(symbol)
        draw=int(hashlib.sha256((str(seed)+':'+trial_id).encode()).hexdigest(),16)%35
        targets.append(dict(trial_id=trial_id,intended=intended,target=draw+(draw>=intended)))
    return dict(subject=subject,session=1,role=role,seed=seed,partition=partition,targets=targets,
                raw_sha256=raw_sha256,protocol_sha256=protocol_sha256,
                status='candidate metadata manifest; does not unlock independent performance')

def validate_evaluation_binding(manifest,protocol_sha256,subject,trial_ids):
    if subject==52 or manifest['subject']!=subject or manifest['role']!='evaluation':
        raise ValueError('eligible sealed evaluation identity required')
    if manifest['protocol_sha256']!=protocol_sha256:
        raise ValueError('protocol hash mismatch')
    if manifest['partition']['evaluation']!=trial_ids:
        raise ValueError('exact evaluation identity/order mismatch')
    from p3a.data.openbmi_partitions import validate_partition
    validate_partition(manifest['partition'])
    return True
