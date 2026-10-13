# Physics-Consistency Slip Estimation for Mecanum-Wheeled Mobile Robots

Reproduction package for the manuscript:

> **Physics-Consistency Slip Estimation for Mecanum-Wheeled Mobile Robots with
> Sparse Weak Labels and a Lightweight Attention Encoder** — Yimeng Chai et al.
> (manuscript under review)

This repository contains the complete simulation pipeline: the domain-randomized
data generator, classical and supervised baselines, the two-stage estimator
(attention encoder + per-sequence physics-consistency refinement), all ablation
and robustness experiments, and the exact archives that generate every number,
table, and figure in the paper. No real-robot hardware is required.

## Requirements
- Python 3.8+, numpy, torch (CPU or CUDA), matplotlib (figures only)
- All seeds are fixed; the pipeline is deterministic on the same machine.

## Structure
- `code/` — simulator, baselines, models, training/evaluation, figures, tables, number audit
- `code/checkpoints/` — trained model weights (attention encoders + supervised references)
- `data/` — generated datasets and all result archives (`.npz`) used in the paper

## Reproduction order (run each command inside `code/`)

| Step | Command | Output | Runtime |
|---|---|---|---|
| 1 | `python mecanum_sim.py` | `data/dataset_v1.npz`, `raw_trajs_v1.npz`, `labels5_v1.npy` | <1 min |
| 2 | `python baselines.py` | `data/baseline_results_v1.npz` | ~1 min |
| 3 | `python train_supervised.py` | `data/supervised_results_v1.npz` | ~1 min |
| 4 | `python run_main_method.py` | `data/main_method_results_v1.npz` (Table 1 main row) | ~7 min |
| 5 | `python run_ablations.py` | `data/ablation_results_v1.npz`, `fig_preds_v1.npz` (Table 2, Figs 3-4) | ~30 min |
| 6 | `python run_multiseed.py` | `data/multiseed_results_v1.npz` | ~15 min |
| 7 | `python run_noise_seeds.py` | `data/noise_multiseed_v1.npz` | ~10 min |
| 8 | `python run_mismatch.py` | `data/mismatch_results_v1.npz` (Fig. S2) | ~5 min |
| 9 | `python run_anchor_sensitivity.py` | `data/anchor_sensitivity_v1.npz` (Fig. S1) | ~50 min |
| 10 | `python run_init_confound.py` | `data/init_confound_v1.npz` | ~6 min |
| 11 | `python fix_table2_init_row.py` | `data/table2_init_matched_v1.npz` | ~3 min |
| 12 | `python run_convergence.py` | `data/convergence_v1.npz` (Fig. S3) | ~10 min |
| 13 | `python stats_summary.py` | `data/stats_summary_v1.json` (p-values, CIs) | <1 min |
| 14 | `python make_figures.py && python make_figures_sci.py` | figures | <1 min |
| 15 | `python make_tables.py` | LaTeX tables | <1 min |
| 16 | `python audit_numbers.py` | number audit (ALL OK expected) | <1 min |

## Paper item -> archive map

| Paper item | File(s) |
|---|---|
| Table 1 (main comparison) | `data/baseline_results_v1.npz`, `supervised_results_v1.npz`, `main_method_results_v1.npz` |
| Table 2 (ablation matrix) | `data/ablation_results_v1.npz`, `table2_init_matched_v1.npz` |
| Figs 1-2 (trajectories, slip zoom) | `data/fig_preds_v1.npz`, `raw_trajs_v1.npz` |
| Figs 3-4 (label curve, noise) | `data/ablation_results_v1.npz` |
| Fig. S1 / S2 / S3 | `data/anchor_sensitivity_v1.npz` / `mismatch_results_v1.npz` / `convergence_v1.npz` |
| Significance tests, CIs, 5-seed stats | `data/stats_summary_v1.json`, `multiseed_results_v1.npz`, `noise_multiseed_v1.npz` |
| Latency measurements | `data/latency_v1.npz` |
| Initialization/anchor confound audit | `data/init_confound_v1.npz` |

## Method summary
The reference budget is 5% of time steps per sequence (15 of 300), drawn with a fixed
per-sequence seed and **used identically at both stages**: Stage 1 supervises the
encoder on exactly these labeled steps (the loss is evaluated only on them, 71,812
parameters), and Stage 2 refines each sequence by minimizing the wheel-plus-IMU
consistency objective (kinematic-intercompatibility residual, gyro alignment,
accelerometer increments, temporal-mean prior) anchored on the same set. The
consistency objective is provably invariant to a two-dimensional global-velocity
gauge, which is why the sparse anchors are necessary; the paper quantifies the
resulting event-region accuracy and its boundaries.

## License
MIT (see `LICENSE`).

## Citation
If you use this code, please cite the manuscript above (DOI to be added upon publication).
