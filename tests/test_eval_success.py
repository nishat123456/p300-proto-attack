"""Fast CPU tests for targeted success at 15 and 5 repetitions (Prompt 10)."""
import numpy as np
import torch

from p3a.data.layout import COMP3
from p3a.eval.success import (
    FLASHES_PER_REP,
    REPS,
    reduced_rep_choice,
    success_all_reps,
    success_for_speller,
    targeted_success,
)
from p3a.models.letter_head import symbol_logits

from tests.eval_helpers import FLASHES_PER_REP as N_PER_REP, make_speller, make_trials, batch


def test_flashes_per_rep_matches_comp3():
    assert FLASHES_PER_REP == N_PER_REP == 12
    assert REPS == (15, 5)


def test_targeted_success_bounds():
    choice = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 9, 2, 9])
    assert targeted_success(choice, target) == 0.5
    assert targeted_success(choice, choice.clone()) == 1.0
    assert np.isnan(targeted_success(torch.tensor([]), torch.tensor([])))


def test_reduced_rep_choice_uses_first_reps_only():
    trials = make_trials(2)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    with torch.no_grad():
        scores = speller(eeg, onsets, groups)["flash_scores"]
    full = reduced_rep_choice(scores, groups, None, 15)
    # Replacing all flashes after the fifth repetition must not change the 5-rep
    # decision, because only the first 60 flashes are read.
    perturbed = scores.clone()
    perturbed[:, 5 * FLASHES_PER_REP:] += 50.0
    five = reduced_rep_choice(perturbed, groups, None, 5)
    assert torch.equal(five, reduced_rep_choice(scores, groups, None, 5))
    assert full.shape == five.shape == (2,)


def test_reduced_rep_choice_rejects_too_many_reps():
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    with torch.no_grad():
        scores = speller(eeg, onsets, groups)["flash_scores"]
    import pytest

    with pytest.raises(ValueError, match="exceed"):
        reduced_rep_choice(scores, groups, None, 99)


def test_success_for_speller_returns_all_and_reduced():
    trials = make_trials(3)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    target = torch.tensor([0, 1, 2], dtype=torch.long)
    full = success_for_speller(speller, eeg, onsets, groups, target, 15)
    five = success_for_speller(speller, eeg, onsets, groups, target, 5)
    assert 0.0 <= full <= 1.0 and 0.0 <= five <= 1.0

    all_reps = success_all_reps(speller, eeg, onsets, groups, target)
    assert set(all_reps) == {15, 5}
    assert all_reps[15] == full and all_reps[5] == five


def test_clean_choice_matches_direct_letter_head():
    """The success module must agree with a direct letter-head decision."""
    trials = make_trials(2)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    with torch.no_grad():
        out = speller(eeg, onsets, groups)
    n = 15 * FLASHES_PER_REP
    expected = symbol_logits(out["flash_scores"][:, :n], groups[:, :n]).argmax(dim=1)
    target = torch.tensor([int(COMP3.index(t.target_symbol)) for t in trials])
    assert success_for_speller(speller, eeg, onsets, groups, target, 15) == \
        targeted_success(expected, target)


def test_large_perturbation_flips_five_rep_choice():
    """A huge delta must change the 5-rep decision on the affine toy speller."""
    trials = make_trials(1)
    speller = make_speller(trials)
    eeg, onsets, groups = batch(trials)
    target = torch.tensor([7], dtype=torch.long)
    before = success_for_speller(speller, eeg, onsets, groups, target, 5)
    from p3a.attacks.baselines import targeted_pgd

    speller.requires_grad_(False)
    delta = targeted_pgd(speller, eeg, onsets, groups, target, 50.0, steps=40)
    after = success_for_speller(speller, eeg + delta, onsets, groups, target, 5)
    assert after >= before
    assert after == 1.0
