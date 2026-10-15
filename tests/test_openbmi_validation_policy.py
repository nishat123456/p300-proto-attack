import sys
import pytest
from scripts import evaluate_openbmi_validation_clean as clean_runner
from scripts import audit_openbmi_attacks as attack_runner


@pytest.mark.parametrize('runner',[clean_runner,attack_runner])
def test_validation_runner_rejects_sealed_before_artifact_read(monkeypatch,runner):
    argv=['runner','--subject','37','--seed','20261013']
    for name in ('checkpoint','references','calibration','user-preprocessor','raw-file','schedule','out-dir','git-commit'):
        argv+=['--'+name,'unread']
    if runner is attack_runner:argv+=['--template','unread','--reference-json','unread']
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(runner.torch,'load',lambda *a,**k:pytest.fail('sealed artifact read'))
    with pytest.raises(ValueError,match='validation subjects'):runner.main()
