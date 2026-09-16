from __future__ import annotations

import numpy as np


def solve_non_exhaustive_assignment(score_matrix: np.ndarray, valid_matrix: np.ndarray) -> list[tuple[int, int]]:
    scores = np.asarray(score_matrix, dtype=np.float64)
    valid = np.asarray(valid_matrix, dtype=bool)
    if scores.shape != valid.shape:
        raise ValueError("score_matrix and valid_matrix must have the same shape")
    n_obs, n_nodes = scores.shape
    if n_obs == 0 or n_nodes == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as exc:
        raise RuntimeError("SciPy is required for Hungarian assignment") from exc
    size = n_obs + n_nodes
    aug = np.full((size, size), -1.0e9, dtype=np.float64)
    aug[:n_obs, :n_nodes][valid] = scores[valid]
    for i in range(n_obs):
        aug[i, n_nodes + i] = 0.0
    for j in range(n_nodes):
        aug[n_obs + j, j] = 0.0
    aug[n_obs:, n_nodes:] = 0.0
    rows, cols = linear_sum_assignment(-aug)
    matches: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r < n_obs and c < n_nodes and valid[r, c]:
            matches.append((r, c))
    return matches
