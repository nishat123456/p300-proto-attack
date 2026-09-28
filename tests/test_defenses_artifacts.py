"""Fast CPU tests for the attack-artifact loader (Prompt 11)."""
import numpy as np
import pandas as pd

from p3a.defenses import artifacts as art

from tests.defense_helpers import write_subject


def test_available_combos_lists_written_specs(tmp_path):
    write_subject(tmp_path, "A", methods=[("pgd", "none", "none", 1.0),
                                          ("tiny_noise", "none", "none", 1.0)],
                  eps=(0.10, 0.20))
    specs = art.available_combos(tmp_path, "A")
    labels = {s.label for s in specs}
    assert labels == {"pgd_none_none", "tiny_noise_none_none"}
    assert {float(s.eps_multiplier) for s in specs} == {0.10, 0.20}
    filtered = art.available_combos(tmp_path, "A", eps_multipliers=[0.20])
    assert all(float(s.eps_multiplier) == 0.20 for s in filtered)


def test_load_combo_alignment_and_eps(tmp_path):
    write_subject(tmp_path, "A", n_trials=5, n_targets=3, methods=[("pgd", "none", "none", 1.0)])
    spec = art.available_combos(tmp_path, "A")[0]
    combo = art.load_combo(tmp_path, "A", spec)
    assert combo.n_trials == 5 and combo.n_targets == 3
    assert combo.eps_raw == 1.0
    inputs = art.combo_to_inputs(combo)
    assert inputs.attacked_erp.shape[0] == 15
    assert inputs.attacked_erp.shape == inputs.clean_duplicate_erp.shape
    assert inputs.genuine_erp.shape == inputs.attacked_erp.shape
    assert inputs.success.all()
    # Every genuine ERP repeats its trial, so unique trials equal n_trials.
    assert len(set(inputs.trial_id.tolist())) == 5


def test_success_mask_respects_manifest_flags(tmp_path):
    write_subject(tmp_path, "A", n_trials=8, n_targets=2,
                  methods=[("pgd", "none", "none", 0.5)],
                  seed=1)
    spec = art.available_combos(tmp_path, "A")[0]
    combo = art.load_combo(tmp_path, "A", spec)
    mask = combo.success_mask()
    assert mask.dtype == bool and mask.shape == (16,)
    # A 50 percent success rate must leave both classes present in 16 draws.
    assert 0 < mask.sum() < mask.size


def test_manifest_failure_mask_all_false(tmp_path):
    write_subject(tmp_path, "A", n_trials=4, n_targets=2,
                  methods=[("pgd", "none", "none", 0.0)])
    spec = art.available_combos(tmp_path, "A")[0]
    combo = art.load_combo(tmp_path, "A", spec)
    assert not combo.success_mask().any()


def test_manifest_alignment_preserves_predrawn_target_order(tmp_path):
    directory = write_subject(
        tmp_path, "A", n_trials=2, n_targets=3,
        methods=[("pgd", "none", "none", 1.0)],
    )
    frame = pd.read_parquet(directory / "attacks.parquet")
    # Encode success by target id, then scramble parquet row order. The loader
    # must restore the NPZ's pre-drawn target order rather than sorting targets.
    frame["success_surrogate_15"] = (frame["target_idx"] % 2).astype(float)
    frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    frame.to_parquet(directory / "attacks.parquet", index=False)
    spec = art.available_combos(tmp_path, "A")[0]
    combo = art.load_combo(tmp_path, "A", spec)
    expected = (combo.attack.target_idx.reshape(-1) % 2).astype(bool)
    np.testing.assert_array_equal(combo.success_mask(), expected)


def test_combine_samples_concatenates(tmp_path):
    write_subject(tmp_path, "A", n_trials=3, n_targets=2,
                  methods=[("pgd", "none", "none", 1.0), ("noise", "none", "none", 1.0)])
    inputs = [art.combo_to_inputs(art.load_combo(tmp_path, "A", s))
              for s in art.available_combos(tmp_path, "A")]
    merged = art.combine_samples(inputs)
    assert merged.attacked_erp.shape[0] == sum(c.attacked_erp.shape[0] for c in inputs)
    assert len(merged.meta["labels"]) == 2


def test_result_dir_layout(tmp_path):
    assert art.result_dir(tmp_path, "a") == tmp_path / "attacks_eval" / "A"
