"""
Smoke tests for torch-dependent helpers (finetuning thresholds, robustness
volume perturbation). The whole module is skipped when torch is unavailable.
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'finetuning'))
sys.path.insert(0, os.path.join(ROOT, 'robustness'))

from finetune_LFM import set_seed, youden_threshold  # noqa: E402
from perturbation import generate_shifts, shift_volume  # noqa: E402


def test_youden_threshold_separates_classes():
    labels = np.array([0] * 50 + [1] * 50)
    probs = np.concatenate([np.linspace(0.0, 0.4, 50),
                            np.linspace(0.6, 1.0, 50)])
    thr = youden_threshold(labels, probs)
    assert 0.4 <= thr <= 0.6


def test_shift_volume_preserves_shape_and_shifts_content():
    vol = np.zeros((8, 16, 16), dtype=np.float32)
    vol[:, :, 4] = 1.0
    out = shift_volume(vol, 2, 0, 0)
    assert out.shape == vol.shape
    # Content moves toward lower x indices by sx; vacated border is zero.
    assert np.all(out[:, :, 2] == 1.0)
    assert np.all(out[:, :, -1] == 0.0)


def test_shift_volume_pad_modes():
    vol = np.ones((8, 16, 16), dtype=np.float32)
    out_edge = shift_volume(vol, 3, -2, 1, pad_mode='edge')
    assert out_edge.shape == vol.shape
    # Edge padding replicates the border value (1.0 everywhere here).
    assert np.all(out_edge == 1.0)
    with pytest.raises(ValueError):
        shift_volume(vol, 1, 0, 0, pad_mode='bogus')


def test_generate_shifts_is_deterministic():
    patients = [{'D': 64}, {'D': 32}, {'D': 48}]
    s1 = generate_shifts(patients, 8, np.random.default_rng(42))
    s2 = generate_shifts(patients, 8, np.random.default_rng(42))
    assert s1 == s2
    zero = generate_shifts(patients, 0, np.random.default_rng(42))
    assert all(s == (0, 0, 0) for s in zero)


def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)
