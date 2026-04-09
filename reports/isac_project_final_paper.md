# Probabilistic Sensing and Waveform Co-Optimization for ISAC Power Allocation: Final Project Report

## Abstract
This project extends a modular integrated sensing and communication (ISAC) power-allocation codebase from a sensing-SNR-only objective to a more realistic sensing-performance formulation. The final system introduces two major capabilities: (1) probabilistic sensing optimization using detection probability (`Pd`) at configurable false-alarm probability (`Pfa`), and (2) joint power-plus-waveform co-optimization through an expanded decision vector. These additions are integrated across objective evaluation, Pareto analysis, scalar solvers, multi-objective NSGA-II search, dynamic time-varying experiments, serialization, visualization, and tests. The resulting framework preserves backward compatibility with legacy SNR-based studies while enabling advanced ISAC experimentation under richer sensing objectives and waveform design degrees of freedom.

## 1. Introduction
Integrated sensing and communication (ISAC) systems must balance communication throughput and sensing quality under shared resource constraints. The original project modeled this trade-off with communication rate and sensing SNR. While useful, sensing SNR alone is not always a task-level metric for detection-oriented sensing, where probability of detection is more meaningful.

This final project therefore aimed to:
1. Support probabilistic sensing metrics directly in optimization.
2. Add waveform/beam-pattern style co-optimization alongside power allocation.
3. Preserve existing pipelines (Pareto sweep, algorithm comparison, dynamic comparison).
4. Keep all optimizers functional under both legacy and extended modes.

## 2. Baseline Formulation (Legacy-Compatible)
The communication utility remains Shannon-style:

`R(p) = sum_k log2(1 + p_k * g_comm,k / N0)`

The baseline sensing quantity remains:

`SNR_sense(p) = (sum_k p_k * g_sense,k) / (K * N0)`

Legacy weighted optimization is still available:

`J = alpha * R + (1 - alpha) * gamma * log10(SNR_sense)`

This mode is retained through `sensing_metric="snr"` for reproducibility of earlier experiments.

## 3. Extended Sensing Objective
### 3.1 Detection-Probability Utility
A probabilistic sensing path was added through `sensing_metric="detection_probability"`. The project computes:

1. Threshold from configured `Pfa`.
2. Integrated sensing SNR using configurable integration gain.
3. Detection probability `Pd` from a Gaussian-tail approximation.

The scalar objective is then:

`J = alpha * R + (1 - alpha) * gamma * U_sense`

where `U_sense` is either `log10(SNR_sense)` (legacy) or `Pd` (new mode).

### 3.2 Reported Metrics
Each solution now reports:
1. Communication rate.
2. Sensing SNR (linear and dB).
3. Detection probability.
4. Sensing utility value used by the optimizer.
5. Name of active sensing metric mode.

## 4. Waveform Co-Optimization
### 4.1 Decision Variable Expansion
When enabled (`waveform_co_optimization=True`), optimization uses:

`d = [p_1 ... p_K, w_1 ... w_K]`

where:
1. `p_k` are per-subcarrier powers.
2. `w_k` is a normalized waveform/beam-profile weight vector.

Constraints:
1. `sum_k p_k <= P_total` with optional per-subcarrier limits.
2. `w_k >= w_min`.
3. `sum_k w_k = K` normalization.

### 4.2 Gain Shaping
Waveform profile modulates effective communication and sensing gains with separate exponents:

`g_comm,k_eff = g_comm,k * w_k^(beta_comm)`

`g_sense,k_eff = g_sense,k * w_k^(beta_sense)`

This allows the user to control how aggressively waveform shaping affects communication and sensing channels.

## 5. System-Level Integration
The extended model was integrated across the full codebase:

1. Objective/config layer:
   - Added sensing-mode and waveform-co-optimization configuration fields.
   - Added decision-space helpers for equal/random initialization and repair.
2. Solvers:
    - Updated GWO, PSO, DE, SA, SFO, POA, and NSGA-II to use the new decision dimension transparently.
   - Solvers now return power allocation plus waveform profile (when enabled).
3. Pareto utilities:
   - Dominance checks now use sensing utility (generic across SNR and `Pd` modes).
4. Experiment runner and JSON summaries:
   - Added waveform serialization.
   - Added dynamic aggregate statistics for mean/std detection probability.
   - Included problem metadata for sensing mode and co-optimization settings.
5. Plotting and scripts:
   - Pareto y-axis now adapts automatically to either sensing SNR or detection probability.
   - Console outputs now include `Pd` in algorithm and dynamic comparison reports.

## 6. Verification and Validation
Validation was performed through:
1. Unit tests: all tests pass, including new coverage for probabilistic detection with waveform co-optimization.
2. End-to-end scripts: Pareto analysis, algorithm comparison, and dynamic comparison run successfully after integration.
3. Feature smoke tests: dedicated runs in detection-probability mode with waveform co-optimization confirmed expected outputs, including non-null waveform profiles and correct sensing metric labeling.

## 7. Practical Impact
The final project transitions the repository from a single sensing proxy toward a more realistic ISAC research platform:

1. Users can optimize for detection relevance (`Pd`) instead of only SNR.
2. Users can study joint resource shaping (power + waveform profile), not only power budgets.
3. Existing experiments remain runnable without migration burden.
4. The framework now supports richer Pareto interpretation under sensing-task-aware objectives.

## 8. Limitations and Future Work
Current probabilistic detection modeling is intentionally lightweight for algorithmic experimentation. Future extensions can include:
1. Task-specific detector families and threshold models.
2. Multi-target and multi-user ISAC formulations.
3. Imperfect CSI and robust/stochastic optimization.
4. Beamforming with explicit array geometry and covariance constraints.
5. Statistical confidence reporting over repeated channel realizations.

## 9. Conclusion
This project successfully implemented the requested extension from SNR-only sensing optimization to probabilistic detection-aware ISAC optimization and added waveform co-optimization in a backward-compatible, modular manner. The resulting codebase is more realistic, more expressive, and better aligned with modern ISAC research directions, while preserving reproducibility of prior baseline experiments.
