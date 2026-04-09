# ISAC Power Allocation

This repository implements a modular codebase for the paper-level ISAC power allocation problem and prioritizes the two strongest peer-review actions:

- explicit Pareto-front analysis instead of only a single weighted sum point
- more realistic, scenario-driven time-varying channel models instead of a purely simplistic fading model

The current implementation now centers probabilistic detection (`Pd`) and joint power-plus-waveform co-optimization as the default sensing design path. It keeps the original Grey Wolf Optimizer (GWO) as a scalarized solver, adds NSGA-II as a reference multi-objective solver for Pareto analysis, and separates channels, objectives, optimizers, plotting, and experiment runners into independent modules.

## Project Layout

```text
isac_power_allocation/
  channels/
    models.py
    scenarios.py
  experiments/
    runner.py
  optimizers/
    baselines.py
    gwo.py
    nsga2.py
  config.py
  constraints.py
  math_utils.py
  objectives.py
  pareto.py
  plotting.py
scripts/
  run_pareto_analysis.py
tests/
  test_channels.py
  test_constraints.py
  test_optimizers.py
```

## What Changed Relative To The Paper Draft

- The original weighted objective is preserved for direct comparability with the paper.
- Pareto analysis is now explicit through:
  - alpha sweeps with GWO
  - a true multi-objective NSGA-II front
  - non-dominated filtering utilities
- The default sensing objective is probabilistic detection (`Pd`) at configurable `Pfa`.
- Joint power and waveform co-optimization is enabled by default so the sensing side of the problem is not limited to power allocation alone.
- Channel realism is improved with standardized-scenario-inspired presets:
  - `UMi_NLOS`
  - `UMa_LOS`
  - `WINNER_A1`
  - `ITU_PedB`
- Channels are frequency-selective and time-varying:
  - path loss
  - log-normal shadowing
  - clustered delay spread
  - Doppler evolution
  - optional LOS/Rician component
- Sensing is modeled as a separate round-trip channel with configurable processing gain, instead of reusing the communication link unchanged.
- Default presets include communication and sensing processing gains so the standardized-scenario-inspired channel remains physically motivated while still producing informative Pareto fronts for algorithm analysis.
- The sensing objective supports probabilistic detection utility (`Pd` at configured `Pfa`) by default, while legacy SNR utility remains available for compatibility.
- Optional co-optimization of power and a normalized waveform/beam-pattern profile is supported through objective config flags.

## Assumptions

The peer review asked for realistic standardized models such as 3GPP / WINNER / ITU scenarios. This repository implements lightweight scenario-inspired approximations suitable for algorithm development and Pareto analysis, not a full standard-compliant channel emulator. The scenario presets expose the main physical knobs the review asked for: path loss severity, shadowing, delay spread, LOS structure, and Doppler.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python .\scripts\run_pareto_analysis.py
```

```powershell
python .\scripts\run_algorithm_comparison.py
```

```powershell
python .\scripts\run_dynamic_algorithm_comparison.py
```

Probabilistic detection mode:

```powershell
python .\scripts\run_algorithm_comparison_pd.py --integration-gain 0.02
python .\scripts\run_pareto_pd.py --integration-gain 0.02
python .\scripts\run_dynamic_pd.py --integration-gain 0.02
```

Disable waveform co-optimization only if you want the legacy power-only variant:

```powershell
python .\scripts\run_algorithm_comparison_pd.py --integration-gain 0.02 --no-waveform
python .\scripts\run_pareto_pd.py --integration-gain 0.02 --no-waveform
python .\scripts\run_dynamic_pd.py --integration-gain 0.02 --no-waveform
```

This generates:

- `outputs/pareto_front.png`
- `outputs/channel_snapshot.png`
- `outputs/pareto_summary.json`
- `outputs/algorithm_comparison.png`
- `outputs/algorithm_comparison_summary.json`
- `outputs/dynamic_algorithm_comparison.png`
- `outputs/dynamic_algorithm_comparison_summary.json`
- `outputs/dynamic_objective_traces.png`
- `outputs/pareto_front_pd*.png`
- `outputs/channel_snapshot_pd*.png`
- `outputs/pareto_summary_pd*.json`
- `outputs/algorithm_comparison_pd*.png`
- `outputs/algorithm_comparison_pd*_summary.json`
- `outputs/dynamic_algorithm_comparison_pd*.png`
- `outputs/dynamic_algorithm_comparison_summary_pd*.json`
- `outputs/dynamic_objective_traces_pd*.png`

## Test

```powershell
python -m unittest discover -s tests -v
```
