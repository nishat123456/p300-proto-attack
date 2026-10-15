import numpy as np
import pytest
from p3a.defenses.network_reference_scores import source_reference_indices


def test_balanced_reproducible_source_only_without_replacement():
    owners=np.repeat(np.arange(1,19),20);labels=np.tile([0]*10+[1]*10,18)
    ix=source_reference_indices(owners,labels,72,7)
    assert len(np.unique(ix))==72 and np.array_equal(ix,source_reference_indices(owners,labels,72,7))
    assert all(np.sum((owners[ix]==s)&(labels[ix]==c))==2 for s in range(1,19) for c in (0,1))
    assert len(source_reference_indices(owners,labels,5000,7))==360
    owners[0]=37
    with pytest.raises(ValueError):source_reference_indices(owners,labels,72,7)


def test_missing_source_class_rejected():
    owners=np.repeat(np.arange(1,19),2);labels=np.tile([0,1],18);labels[1]=0
    with pytest.raises(ValueError):source_reference_indices(owners,labels,36,7)
