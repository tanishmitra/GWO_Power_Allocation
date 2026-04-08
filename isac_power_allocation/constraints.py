"""Feasibility and projection helpers for power allocations."""

from __future__ import annotations

import numpy as np


def _project_to_simplex(values: np.ndarray, total_sum: float) -> np.ndarray:
    if total_sum <= 0:
        return np.zeros_like(values)
    if values.sum() <= total_sum and np.all(values >= 0):
        return values.copy()

    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values)
    rho_candidates = sorted_values - (cumulative - total_sum) / (np.arange(len(values)) + 1)
    rho = int(np.nonzero(rho_candidates > 0)[0][-1])
    theta = (cumulative[rho] - total_sum) / float(rho + 1)
    return np.maximum(values - theta, 0.0)


def project_power_allocation(
    values: np.ndarray,
    total_power_w: float,
    per_subcarrier_max_power_w: float | None = None,
) -> np.ndarray:
    projected = np.maximum(np.asarray(values, dtype=float), 0.0)
    dimension = projected.size

    if dimension == 0:
        return projected

    if per_subcarrier_max_power_w is None:
        if projected.sum() == 0:
            return np.full(dimension, total_power_w / dimension)
        return projected * (total_power_w / projected.sum())

    feasible_total = dimension * per_subcarrier_max_power_w
    capped_total = min(total_power_w, feasible_total)

    projected = np.minimum(projected, per_subcarrier_max_power_w)
    if projected.sum() > capped_total:
        return _project_to_simplex(projected, capped_total)

    remaining = capped_total - projected.sum()
    tolerance = 1e-12

    while remaining > tolerance:
        free_mask = projected < (per_subcarrier_max_power_w - tolerance)
        if not np.any(free_mask):
            break

        weights = projected[free_mask]
        if weights.sum() <= tolerance:
            weights = np.ones(free_mask.sum(), dtype=float)

        increments = remaining * weights / weights.sum()
        headroom = per_subcarrier_max_power_w - projected[free_mask]
        delta = np.minimum(increments, headroom)
        projected[free_mask] += delta

        consumed = float(delta.sum())
        if consumed <= tolerance:
            projected[free_mask] += np.minimum(headroom, remaining / free_mask.sum())
            break

        remaining = capped_total - projected.sum()

    if projected.sum() == 0:
        uniform = np.full(dimension, capped_total / dimension)
        return np.minimum(uniform, per_subcarrier_max_power_w)

    return projected
