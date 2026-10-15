import sys
import pytest
from scripts import evaluate_openbmi_validation_clean as runner


def test_validation_runner_rejects_sealed_before_artifact_read(monkeypatch):
    argv=['runner','--subject','37','--seed','20261013']
    for name in ('checkpoint','references','calibration','user-preprocessor','raw-file','schedule','out-dir','git-commit'):
        argv+=['--'+name,'unread']
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(runner.torch,'load',lambda *a,**k:pytest.fail('sealed artifact read'))
    with pytest.raises(ValueError,match='validation subjects'):runner.main()
