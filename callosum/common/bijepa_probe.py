"""Closed-form linear probes of the Bi-JEPA partner latents (PLAN.md step 3.3 diagnostics).

Pure PyTorch, no simulator import, so `tests/test_wrench_ppo_bijepa.py` runs on macOS/CI when
torch is installed. The trainer uses these numbers for the report's "prediction quality" column:
the auxiliary loss is a latent MSE that is hard to read in isolation, while a held-out linear
probe `ẑ_partner -> partner's commanded wrench` answers the question the thesis asks -- does the
predicted latent actually carry the partner's current action?

The probe is a diagnostic only: no gradients, no parameters of any optimizer, and it never
feeds back into training.
"""

import torch


def ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge: float = 1e-2,
    test_fraction: float = 0.25,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a closed-form ridge regression from `features` to `targets`; report held-out error.

    A deterministic `test_fraction` share of the rows is held out (the seed fixes the row
    permutation, so the same batch gives the same numbers), the ridge solution is fitted on the
    rest and scored on the held-out rows. `baseline_mse` is the error of predicting the training
    target mean, so `r2 = 1 - mse / baseline_mse` is the usual coefficient of determination
    (negative when the probe is worse than the mean, i.e. the latent carries no linear signal).

    Args:
        features: `(N, d)` probe inputs, e.g. the predicted partner latent `ẑ_j`.
        targets: `(N, m)` probe targets, e.g. the partner's 6D commanded wrench or its latent.
        ridge: L2 coefficient added to the Gram matrix of the standardised (centred,
            unit-variance) features; it only makes the solve stable when `d` is close to the
            row count.
        test_fraction: Held-out share of the rows, strictly between 0 and 1.
        seed: Seed of the row permutation that selects the held-out rows.

    Returns:
        `{"mse", "r2", "baseline_mse", "n_train", "n_test"}` (counts as floats); `r2` is 1.0
        when the held-out targets are constant and the probe is exact, 0.0 when they are
        constant and it is not.

    Raises:
        ValueError: if the shapes disagree, fewer than two rows are given, or a scalar argument
            is outside its valid range.
    """
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError(
            "ridge_probe expects 2-D (N, d) features and (N, m) targets, got "
            f"{tuple(features.shape)} and {tuple(targets.shape)}"
        )
    if features.shape[0] != targets.shape[0]:
        raise ValueError(
            f"ridge_probe row counts differ: features {features.shape[0]}, "
            f"targets {targets.shape[0]}"
        )
    if features.shape[0] < 2:
        raise ValueError(f"ridge_probe needs at least 2 rows, got {features.shape[0]}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")

    x = features.detach().to(torch.float64).cpu()
    y = targets.detach().to(torch.float64).cpu()
    n = x.shape[0]
    n_test = min(max(round(n * test_fraction), 1), n - 1)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    test_idx, train_idx = order[:n_test], order[n_test:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_mean, y_mean = x_train.mean(dim=0), y_train.mean(dim=0)
    # Standardise the features (per-dimension train std) before the solve, so `ridge` means
    # the same thing whatever the latent scale is; a shrinking latent (collapse) must not be
    # flattened by an accidentally over-strong regulariser.
    x_scale = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_centred, y_centred = (x_train - x_mean) / x_scale, y_train - y_mean
    gram = x_centred.T @ x_centred
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=x.dtype)
    weights = torch.linalg.solve(gram, x_centred.T @ y_centred)
    prediction = ((x[test_idx] - x_mean) / x_scale) @ weights + y_mean
    mse = float((prediction - y[test_idx]).pow(2).mean())
    baseline_mse = float((y[test_idx] - y_mean).pow(2).mean())
    if baseline_mse <= 1e-12:
        r2 = 1.0 if mse <= 1e-12 else 0.0
    else:
        r2 = 1.0 - mse / baseline_mse
    return {
        "mse": mse,
        "r2": r2,
        "baseline_mse": baseline_mse,
        "n_train": float(train_idx.numel()),
        "n_test": float(test_idx.numel()),
    }
