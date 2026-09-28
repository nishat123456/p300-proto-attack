"""Kaggle kernel: the Comp III A <-> B kill test (Prompt 12).

Flow, for the requested subjects and methods, writing under
``/kaggle/working/out/kill_test``:

1. locate the attached code bundle and the ``train_decoders`` kernel output,
   copy the bundle and the decoder checkpoints into the writable working dir and
   ``pip install -e`` the bundle,
2. build the same-subject and cross-subject prototype banks,
3. train every requested attack generator and the Tiny Noise templates, skipping
   any checkpoint that already exists so a restarted run resumes,
4. run ``scripts/evaluate_attacks.py`` for the requested methods and
   ``scripts/run_detectors.py`` when the full method set is present,
5. print elapsed time per stage.

Sharding: ``--shard-index I --shard-count N`` splits the work into N shards so
five Kaggle accounts can run in parallel; each shard trains only its method
families and evaluates only its methods plus the cheap ``noise`` and ``pgd``
baselines (which the defense pool needs). Because the genuine-vs-induced defense
is trained on {pgd, target_only, noise} and evaluated on every method, the
detector stage is skipped on a partial shard and runs once over the merged
``attacks_eval`` tree. ``--subject`` and ``--methods`` override the shard
directly. With no shard flags the default covers the full registered job: both
subjects, every method, and the detector stage.

The kernel sources are attached read-only under ``/kaggle/input``; all writes go
to ``/kaggle/working``. No credentials are read, printed or written.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

INPUT_ROOT = Path(os.environ.get("P3A_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("P3A_WORK_ROOT", "/kaggle/working"))
CODE_DIR = WORK_ROOT / "code"
OUT_DIR = WORK_ROOT / "out" / "kill_test"
LOG_PATH = OUT_DIR / "kill_test.log"
TRACEBACK_PATH = OUT_DIR / "failure_traceback.txt"
# The per-account shard is written next to run.py before pushing (Kaggle script
# kernels take no CLI arguments). Kaggle copies the job directory, so this file
# travels with the kernel.
SHARD_FILE = Path(__file__).resolve().parent / "shard.json"
IGNORE_NAMES = {"__pycache__", ".pytest_cache", ".git", ".ipynb_checkpoints"}

SUBJECTS = ("A", "B")
# Methods that need no trained generator.
CORE_METHODS = ("noise", "pgd")
# The one method whose templates are trained separately.
TINY_METHOD = "tiny_noise"
# Methods trained through the prototype-guided generator path.
GENERATOR_METHODS = ("target_only", "proto_random", "proto_medoid",
                     "proto_nearest", "proto_wrong")
GENERATORS = ("signal", "causal")
ALL_METHODS = CORE_METHODS + (TINY_METHOD,) + GENERATOR_METHODS
# Proto nearest is also trained same-subject for the upper bound.
SAME_SUBJECT_METHODS = ("proto_nearest",)
# Method families used to build shards of comparable cost.
_FAMILIES = (
    ("target_only",),
    ("proto_random", "proto_medoid"),
    ("proto_nearest", "proto_wrong"),
    (TINY_METHOD,),
)


# --------------------------------------------------------------------------
# Bundle and decoder discovery
# --------------------------------------------------------------------------

def _tee(message: str, handle=None) -> None:
    print(message, flush=True)
    if handle is not None:
        handle.write(message + "\n")
        handle.flush()


def find_code_bundle(input_root: Path = INPUT_ROOT) -> Path:
    """Find the attached bundle: a directory whose pyproject declares `p3a`."""
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    candidates = []
    for meta in sorted(input_root.rglob("pyproject.toml")):
        text = meta.read_text(errors="replace")
        candidates.append((meta.parent, "[project]" in text and "p3a" in text))
    if not candidates:
        raise FileNotFoundError(
            f"no pyproject.toml found under {input_root}; "
            "is the p300-proto-attack-code dataset attached?"
        )
    preferred = [parent for parent, is_p3a in candidates if is_p3a]
    return (preferred or [parent for parent, _ in candidates])[0]


def copy_bundle(src: Path, dest: Path = CODE_DIR) -> Path:
    """Copy the bundle to a writable location, replacing any previous copy."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*IGNORE_NAMES))
    return dest


def install_editable(code_dir: Path = CODE_DIR) -> None:
    """pip install -e the copied bundle so `import p3a` resolves."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(code_dir)],
        check=True,
    )


def find_decoder_dir(input_root: Path = INPUT_ROOT) -> Path:
    """Locate the train_decoders kernel output directory of checkpoints.

    The attached kernel output is read-only under /kaggle/input. Prefer a
    directory literally named ``decoders``; otherwise take the directory under
    which the most decoder checkpoints (``*.pt`` / ``*.joblib`` with a metrics
    sidecar) live.
    """
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {input_root}")
    named = [p for p in input_root.rglob("decoders") if p.is_dir()]
    if named:
        return named[0]
    best, best_count = None, 0
    for metrics in input_root.rglob("*.metrics.json"):
        count = len(list(metrics.parent.glob("*.pt"))) + len(list(metrics.parent.glob("*.joblib")))
        if count > best_count:
            best, best_count = metrics.parent, count
    if best is None:
        raise FileNotFoundError(
            "no decoder checkpoints found under /kaggle/input; is the "
            "p3a-train-decoders kernel output attached via kernel_sources?"
        )
    return best


def copy_decoders(src: Path, dest: Path = OUT_DIR / "decoders") -> Path:
    """Copy decoder checkpoints and metrics into the writable output dir."""
    src, dest = Path(src), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*")):
        if path.is_file() and path.suffix in (".pt", ".joblib", ".json"):
            shutil.copy2(path, dest / path.name)
    return dest


# --------------------------------------------------------------------------
# Sharding plan
# --------------------------------------------------------------------------

def plan_units(subjects=SUBJECTS, families=_FAMILIES):
    """The (subject, method-family) work units, one per family per subject."""
    units = []
    for subject in subjects:
        for family in families:
            units.append({"subject": str(subject), "methods": tuple(family)})
    return units


def shard_plan(shard_count: int, subjects=SUBJECTS, families=_FAMILIES):
    """Partition the work units into ``shard_count`` round-robin shards.

    ``shard_count`` must be at least one and no larger than the number of units.
    Each returned shard is a list of coherent ``{"subject", "methods"}`` units
    (a subset of the units given by ``plan_units``), so a shard only ever
    evaluates methods whose generators belong to the same subject. Every unit
    appears in exactly one shard, so the shards together cover the full job.
    """
    units = plan_units(subjects, families)
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_count > len(units):
        raise ValueError(
            f"shard_count {shard_count} exceeds the {len(units)} available work units"
        )
    buckets = [[] for _ in range(shard_count)]
    for i, unit in enumerate(units):
        buckets[i % shard_count].append(unit)
    return buckets


def shard_subjects(bucket) -> list[str]:
    """Sorted subjects present in one shard bucket."""
    return sorted({u["subject"] for u in bucket})


def shard_methods(bucket, subject) -> list[str]:
    """The shard's methods for one subject, in registration order."""
    wanted = {m for u in bucket if u["subject"] == subject for m in u["methods"]}
    return [m for m in ALL_METHODS if m in wanted]


def eval_methods_for(methods) -> list[str]:
    """The methods evaluate_attacks needs for a shard: the shard plus the core.

    ``noise`` and ``pgd`` are cheap and always evaluated because the defense
    training pool is {pgd, target_only, noise}.
    """
    wanted = set(CORE_METHODS) | {str(m) for m in methods}
    return [m for m in ALL_METHODS if m in wanted]


# --------------------------------------------------------------------------
# Stage commands
# --------------------------------------------------------------------------

def _run(cmd, cwd: Path, handle) -> int:
    """Run a command, streaming combined stdout/stderr, and return its code."""
    _tee("running: " + " ".join(str(c) for c in cmd), handle)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _tee(line.rstrip("\n"), handle)
    return proc.wait()


def generator_ckpt(out_dir: Path, subject: str, method: str, generator: str,
                   bank: str, seed: int) -> Path:
    return Path(out_dir) / f"{subject}_{method}_{generator}_{bank}_s{seed}.pt"


def tiny_ckpt(out_dir: Path, subject: str, mult: float, seed: int) -> Path:
    return Path(out_dir) / f"{subject}_tiny_noise_eps{float(mult):.2f}_s{seed}.pt"


def build_banks(code_dir: Path, data_root: Path, labels_dir: Path,
                decoder_dir: Path, out_dir: Path, subjects, seed: int,
                device: str, handle) -> int:
    """Build prototype banks and write bank_report.csv. Non-fatal on failure.

    Bank building is a report artifact; if the raw data for the other subject is
    missing, the failure is logged and the run continues so the attacks and
    detectors still proceed.
    """
    script = code_dir / "scripts" / "bank_report.py"
    rc = _run([
        sys.executable, script,
        "--data-root", data_root, "--labels-dir", labels_dir,
        "--checkpoint-dir", decoder_dir, "--out-dir", out_dir,
        "--model", "eegnet", "--device", device, "--seed", seed,
        "--subjects", *[str(s) for s in subjects],
    ], cwd=code_dir, handle=handle)
    if rc != 0:
        _tee(f"bank_report.py exited with code {rc}; continuing without banks", handle)
    return rc


def train_generators(code_dir: Path, data_root: Path, labels_dir: Path,
                     decoder_dir: Path, out_dir: Path, plan: dict,
                     seed: int, device: str, handle, force: bool = False) -> int:
    """Train every requested generator, checkpointing and resuming per model.

    ``plan`` maps a subject to its method list; only methods that need a trained
    generator are trained here.
    """
    script = code_dir / "scripts" / "train_attack.py"
    trained = 0
    for subject, methods in plan.items():
        surrogate = decoder_dir / f"{subject}_eegnet_s{seed}.pt"
        if not surrogate.is_file():
            raise FileNotFoundError(f"missing surrogate checkpoint {surrogate}")
        for method in methods:
            if method not in GENERATOR_METHODS:
                continue
            banks = ["cross_subject"]
            if method in SAME_SUBJECT_METHODS:
                banks.append("same_subject")
            for bank in banks:
                for generator in GENERATORS:
                    ckpt = generator_ckpt(out_dir, subject, method, generator, bank, seed)
                    if ckpt.is_file() and not force:
                        _tee(f"skip existing generator {ckpt.name}", handle)
                        continue
                    rc = _run([
                        sys.executable, script,
                        "--data-root", data_root, "--labels-dir", labels_dir,
                        "--subject", subject, "--surrogate", surrogate,
                        "--method", method, "--generator", generator,
                        "--bank", bank, "--seed", seed,
                        "--out-dir", out_dir, "--device", device,
                    ], cwd=code_dir, handle=handle)
                    if rc != 0:
                        _tee(f"train_attack failed for {ckpt.name} rc={rc}", handle)
                        return rc
                    trained += 1
    _tee(f"trained {trained} new generator(s)", handle)
    return 0


def train_tiny_noise(code_dir: Path, data_root: Path, labels_dir: Path,
                     decoder_dir: Path, out_dir: Path, plan: dict, eps,
                     seed: int, device: str, handle, force: bool = False) -> int:
    """Train the Tiny Noise templates, one per epsilon, resuming per template."""
    script = code_dir / "scripts" / "train_tiny_noise.py"
    for subject, methods in plan.items():
        if TINY_METHOD not in methods:
            continue
        surrogate = decoder_dir / f"{subject}_eegnet_s{seed}.pt"
        if not surrogate.is_file():
            raise FileNotFoundError(f"missing surrogate checkpoint {surrogate}")
        needed = [e for e in eps
                  if force or not tiny_ckpt(out_dir, subject, e, seed).is_file()]
        if not needed:
            _tee(f"skip existing tiny noise templates for {subject}", handle)
            continue
        rc = _run([
            sys.executable, script,
            "--data-root", data_root, "--labels-dir", labels_dir,
            "--subject", subject, "--surrogate", surrogate,
            "--seed", seed, "--out-dir", out_dir, "--device", device,
        ], cwd=code_dir, handle=handle)
        if rc != 0:
            _tee(f"train_tiny_noise failed for {subject} rc={rc}", handle)
            return rc
    return 0


def run_evaluate(code_dir: Path, data_root: Path, labels_dir: Path,
                 decoder_dir: Path, out_dir: Path, plan: dict,
                 seed: int, device: str, handle, force: bool = False) -> int:
    """Run evaluate_attacks.py once per subject for that subject's methods.

    ``out_dir`` is the kill-test root: generators live under ``<root>/attacks``,
    Tiny Noise templates under ``<root>/tiny_noise``, and the attack artifacts go
    to ``<root>/attacks_eval``, which is what ``run_detectors`` reads. Core
    methods (noise, pgd) are added per subject because the defense pool needs
    them.
    """
    script = code_dir / "scripts" / "evaluate_attacks.py"
    for subject, methods in plan.items():
        subject_methods = eval_methods_for(methods)
        rc = _run([
            sys.executable, script,
            "--data-root", data_root, "--labels-dir", labels_dir,
            "--subject", subject,
            "--surrogate", str(decoder_dir),
            "--victim-dir", str(decoder_dir),
            "--generator-dir", str(out_dir / "attacks"),
            "--tiny-noise-dir", str(out_dir / "tiny_noise"),
            "--out-dir", str(out_dir / "attacks_eval"),
            "--seed", seed, "--device", device,
            "--methods", *subject_methods,
        ] + (["--force"] if force else []), cwd=code_dir, handle=handle)
        if rc != 0:
            _tee(f"evaluate_attacks.py failed for {subject} rc={rc}", handle)
            return rc
    return 0


def run_detectors(code_dir: Path, results_root: Path, subjects, seed: int,
                  device: str, handle) -> int:
    """Run run_detectors.py for each subject over the merged attack tree."""
    script = code_dir / "scripts" / "run_detectors.py"
    for subject in subjects:
        rc = _run([
            sys.executable, script,
            "--subject", str(subject),
            "--results-root", str(results_root),
            "--seed", seed, "--device", device,
        ], cwd=code_dir, handle=handle)
        if rc != 0:
            _tee(f"run_detectors.py failed for {subject} rc={rc}", handle)
            return rc
    return 0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", nargs="+", default=None,
                   help="restrict to these subjects (default: A and B)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="restrict to these methods (default: every registered method)")
    p.add_argument("--shard-index", type=int, default=None,
                   help="which shard this account runs (0-based)")
    p.add_argument("--shard-count", type=int, default=None,
                   help="split the job into N shards for N Kaggle accounts")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="retrain and reevaluate even when checkpoints exist")
    p.add_argument("--skip-banks", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--detectors", choices=("auto", "always", "never"), default="auto",
                   help="auto runs detectors only for the full method set")
    args, _unknown = p.parse_known_args(argv)
    # A Kaggle script kernel takes no CLI arguments, so the shard can be selected
    # by env vars or by a shard.json written next to run.py before pushing.
    # Explicit flags always win over both.
    _apply_env(args)
    if (args.shard_index is None) != (args.shard_count is None):
        raise SystemExit("--shard-index and --shard-count must be given together")
    return args


def load_shard_file(path: Path = None) -> dict:
    """Read a per-account shard selection from ``shard.json`` next to run.py.

    Recognized keys: ``shard_index``, ``shard_count``, ``subjects`` (list or
    comma string) and ``methods`` (list or comma string). A missing file returns
    an empty dict, which selects the full default job. This file carries no
    credentials and is only a selection record.
    """
    path = Path(path) if path is not None else SHARD_FILE
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("shard.json must contain a JSON object")
    return payload


def _split_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [v for v in value.replace(",", " ").split() if v]
    return [str(v) for v in value]


def _apply_env(args) -> None:
    """Fill unset args from shard.json, then from the P3A_* environment vars."""
    shard = load_shard_file()
    if args.shard_index is None and shard.get("shard_index") is not None:
        args.shard_index = int(shard["shard_index"])
    if args.shard_count is None and shard.get("shard_count") is not None:
        args.shard_count = int(shard["shard_count"])
    if args.subject is None and shard.get("subjects") is not None:
        args.subject = _split_list(shard["subjects"])
    if args.methods is None and shard.get("methods") is not None:
        args.methods = _split_list(shard["methods"])

    if args.shard_index is None and os.environ.get("P3A_SHARD_INDEX"):
        args.shard_index = int(os.environ["P3A_SHARD_INDEX"])
    if args.shard_count is None and os.environ.get("P3A_SHARD_COUNT"):
        args.shard_count = int(os.environ["P3A_SHARD_COUNT"])
    if args.subject is None and os.environ.get("P3A_SUBJECTS"):
        args.subject = _split_list(os.environ["P3A_SUBJECTS"])
    if args.methods is None and os.environ.get("P3A_METHODS"):
        args.methods = _split_list(os.environ["P3A_METHODS"])


def resolve_plan(args) -> tuple[dict, bool]:
    """Resolve a per-subject method map and whether the detector stage runs.

    The returned mapping is ``{subject: [methods]}``. Explicit ``--subject`` /
    ``--methods`` produce the same method list for every requested subject.
    Otherwise a shard bucket selects coherent per-subject method sets from the
    round-robin plan. The detector stage runs when the union of the plan's
    methods is the full registered set (or ``--detectors always``), because the
    defense is trained on pooled attack families and evaluated on all of them
    at once.
    """
    full_methods = list(ALL_METHODS)
    if args.shard_count is not None:
        buckets = shard_plan(args.shard_count)
        if not (0 <= args.shard_index < len(buckets)):
            raise SystemExit(f"shard-index must be in [0, {len(buckets) - 1}]")
        bucket = buckets[args.shard_index]
        if args.subject is not None:
            subjects = [str(s).upper() for s in args.subject]
            plan = {s: list(args.methods or full_methods) for s in subjects}
        elif args.methods is not None:
            plan = {s: list(args.methods) for s in shard_subjects(bucket)}
        else:
            plan = {s: shard_methods(bucket, s) for s in shard_subjects(bucket)}
    else:
        subjects = [str(s).upper() for s in (args.subject or SUBJECTS)]
        plan = {s: list(args.methods or full_methods) for s in subjects}

    for subject, methods in plan.items():
        unknown = [m for m in methods if m not in ALL_METHODS]
        if unknown:
            raise SystemExit(f"unknown method(s) for {subject}: {unknown}; "
                             f"expected {full_methods}")
    all_planned = {m for methods in plan.values() for m in methods}
    run_det = (
        args.detectors == "always"
        or (args.detectors == "auto" and all_planned == set(full_methods))
    )
    return plan, run_det


def _finished_outputs(out_dir: Path) -> list[str]:
    """Names of files that survived, so a failure is inspectable."""
    if not out_dir.is_dir():
        return []
    return sorted(
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and p.name not in {"kill_test.log", "failure_traceback.txt"}
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with LOG_PATH.open("w") as handle:
        try:
            plan, run_det = resolve_plan(args)
            subjects = sorted(plan)
            _tee(f"plan: {plan} detectors={run_det}", handle)

            bundle = find_code_bundle()
            _tee(f"found code bundle at {bundle}", handle)
            copy_bundle(bundle, CODE_DIR)
            install_editable(CODE_DIR)

            decoder_src = find_decoder_dir()
            _tee(f"found decoder checkpoints at {decoder_src}", handle)
            decoder_dir = copy_decoders(decoder_src)

            labels_dir = CODE_DIR / "data" / "raw" / "comp3"
            gen_methods = {
                s: [m for m in ms if m in GENERATOR_METHODS] for s, ms in plan.items()
            }
            tiny_methods = {s: ms for s, ms in plan.items() if TINY_METHOD in ms}

            stages = []
            if not args.skip_banks:
                stages.append(("banks", lambda: build_banks(
                    CODE_DIR, INPUT_ROOT, labels_dir, decoder_dir,
                    OUT_DIR / "banks", subjects, args.seed, args.device, handle,
                )))
            if not args.skip_train:
                if any(gen_methods.values()):
                    stages.append(("generators", lambda: train_generators(
                        CODE_DIR, INPUT_ROOT, labels_dir, decoder_dir,
                        OUT_DIR / "attacks", gen_methods, args.seed,
                        args.device, handle, force=args.force,
                    )))
                if tiny_methods:
                    stages.append(("tiny_noise", lambda: train_tiny_noise(
                        CODE_DIR, INPUT_ROOT, labels_dir, decoder_dir,
                        OUT_DIR / "tiny_noise", tiny_methods, _default_eps(),
                        args.seed, args.device, handle, force=args.force,
                    )))
            if not args.skip_eval:
                stages.append(("evaluate", lambda: run_evaluate(
                    CODE_DIR, INPUT_ROOT, labels_dir, decoder_dir,
                    OUT_DIR, plan, args.seed, args.device, handle, force=args.force,
                )))
            if run_det:
                stages.append(("detectors", lambda: run_detectors(
                    CODE_DIR, OUT_DIR, subjects, args.seed, args.device, handle,
                )))

            for name, stage in stages:
                stage_started = time.time()
                _tee("=" * 60, handle)
                _tee(f"stage: {name}", handle)
                rc = stage()
                _tee(f"stage {name} finished in {time.time() - stage_started:.1f}s "
                     f"(rc={rc})", handle)
                if rc != 0:
                    raise RuntimeError(f"stage {name} failed with exit code {rc}")

            _tee(f"kill test finished in {time.time() - started:.1f}s", handle)
            return 0
        except Exception:  # noqa: BLE001 - persist traceback, do not lose outputs
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            handle.write(tb + "\n")
            handle.flush()
            TRACEBACK_PATH.write_text(tb)
            finished = _finished_outputs(OUT_DIR)
            _tee(f"preserved {len(finished)} finished output file(s):", handle)
            for name in finished[:50]:
                _tee(f"  {name}", handle)
            return 1


def _default_eps():
    """The pre-registered epsilon multipliers, imported without p3a at module load."""
    try:
        from p3a.attacks.train import EPS_MULTIPLIERS
        return list(EPS_MULTIPLIERS)
    except Exception:  # noqa: BLE001 - fall back to the frozen grid
        return [0.05, 0.10, 0.20, 0.30, 0.50]


if __name__ == "__main__":
    raise SystemExit(main())
