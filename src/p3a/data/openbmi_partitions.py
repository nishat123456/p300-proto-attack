"""Candidate partitions only; no frozen-file generation on import."""
import numpy as np


def candidate_character_partition(train_ids,test_ids,seed):
    if len(train_ids)!=33 or len(test_ids)!=36:
        raise ValueError('documented33 train/36 test characters required')
    if len(set(train_ids+test_ids))!=69:
        raise ValueError('unique disjoint official phase IDs required')
    order=np.random.default_rng(seed).permutation(33)
    result={'normalization':sorted(train_ids[i] for i in order[:13]),
            'calibration':sorted(train_ids[i] for i in order[13:]),
            'evaluation':list(test_ids)}
    validate_partition(result)
    return result


def validate_partition(partition):
    required={'normalization':13,'calibration':20,'evaluation':36}
    if set(partition)!=set(required): raise ValueError('unexpected partition roles')
    all_ids=[]
    for key,n in required.items():
        ids=partition[key]
        if len(ids)!=n or not all(isinstance(i,str) and i for i in ids):
            raise ValueError('invalid role count or identity')
        all_ids.extend(ids)
    if len(set(all_ids))!=len(all_ids): raise ValueError('role leakage or duplicate IDs')


def candidate_subject_role(subject):
    if not isinstance(subject,int) or not 1<=subject<=54:
        raise ValueError('documented subject ID1..54 required')
    return 'development' if subject<=18 else 'validation' if subject<=36 else 'evaluation'
