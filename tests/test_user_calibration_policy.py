import pytest
from scripts.calibrate_openbmi_user import validate_calibration_subject


def test_calibration_cannot_unlock_sealed_subjects():
    for subject in (19,36):validate_calibration_subject(subject)
    for subject in (1,18,37,52,54,0,55):
        with pytest.raises(ValueError):validate_calibration_subject(subject)
