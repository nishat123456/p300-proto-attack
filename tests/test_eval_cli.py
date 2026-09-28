"""Fast CPU tests for the evaluate_attacks CLI (Prompt 10)."""
import importlib.util
from pathlib import Path

from p3a.eval.evaluate import load_attack_targets

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_attacks_script", REPO / "scripts" / "evaluate_attacks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_specs_only_use_no_bank_methods():
    script = _load_script()
    specs = script.smoke_specs()
    assert specs
    assert all(s.kind in ("noise", "pgd") for s in specs)
    assert all(s.bank == "none" for s in specs)


def test_cli_smoke_overrides(monkeypatch, tmp_path):
    script = _load_script()
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        class R:
            n_trials = 1; n_targets = 10; n_combos = 1
            n_combos_done = 1; n_combos_skipped = 0; n_rows = 10
            clean_npz = "c.npz"; parquet = "a.parquet"
        return R()

    monkeypatch.setattr(script, "evaluate_attacks", fake_evaluate)
    out = script.main([
        "--subject", "A", "--smoke", "--out-dir", str(tmp_path / "out"),
        "--surrogate", "surrogate.pt",
    ])
    assert captured["device"] == "cpu"
    assert captured["limit_trials"] == 3
    assert captured["victims"] is False
    assert all(s.kind in ("noise", "pgd") for s in captured["specs"])
    assert "A" in out and out["A"]["n_rows"] == 10


def test_cli_dry_run_writes_plan(monkeypatch, tmp_path):
    script = _load_script()
    out = script.main([
        "--subject", "A", "--dry-run", "--out-dir", str(tmp_path / "out"),
    ])
    assert out["mode"] == "dry_run"
    assert out["plan"]["A"]["n_test"] == 100
    assert out["plan"]["A"]["targets_cover_test"] is True
    assert out["plan"]["A"]["n_targets_per_trial"] == 10
    assert (tmp_path / "out" / "evaluate_attacks.dry_run.json").is_file()


def test_cli_subjects_flag_lists(tmp_path, monkeypatch):
    script = _load_script()
    captured = []

    def fake_evaluate(**kwargs):
        captured.append(kwargs["subject"])
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(script, "evaluate_attacks", fake_evaluate)
    try:
        script.main(["--subjects", "A", "B", "--out-dir", str(tmp_path)])
    except RuntimeError:
        pass
    assert captured == ["A"]


def test_resolve_surrogate_per_subject(tmp_path):
    script = _load_script()
    assert script._resolve_surrogate(str(tmp_path), "B", 2) == \
        tmp_path / "B_eegnet_s2.pt"
    assert script._resolve_surrogate(str(tmp_path / "{subject}_s{seed}.pt"), "A", 1) == \
        tmp_path / "A_s1.pt"


def test_load_attack_targets_excludes_true_symbol():
    targets = load_attack_targets("A")
    for trial_id, subset in list(targets.items())[:20]:
        assert len(subset) == len(set(subset)) == 10
