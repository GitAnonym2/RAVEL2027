# Synthetic mechanism-discrimination benchmark (Figure 2)

This experiment implements the reciprocal synthetic test described in the submitted paper. The generative truth contains two delayed components. RAVEL is fit with `M=2`; the matched alternative uses two static-gain components with event-time-varying delays; ADM is allowed to select its across-region latent dimension from `K in {1,2,3}` using its native validation marginal likelihood.

## Run RAVEL and the matched dynamic-delay comparator

From the repository root:

```bash
python experiments/synthetic_mechanism_discrimination/run_ravel.py
```

This regenerates the two worlds for seeds `271101`, `271102`, and `271103`, performs RAVEL fitting/ablations, and writes outputs under `results/` (ignored by git). A single-seed smoke run is:

```bash
python experiments/synthetic_mechanism_discrimination/run_ravel.py --seeds 271101
```

## Optional ADM baseline

The ADM implementation is third-party code and is **not vendored** in this repository. After obtaining the authors' implementation, point the wrapper at that checkout:

```bash
python experiments/synthetic_mechanism_discrimination/run_adm.py \
  --adm-repo /path/to/Adaptive-Delay-Model \
  --seeds 271101 271102 271103 --device cuda
```

The wrapper uses native ADM fitting/model selection, then converts the selected model's complete-target source-only prediction to the frozen RAVEL spectral evaluator. No common TEST quantity is used to select `K`.

## Summarize or reproduce the submitted figure

```bash
python experiments/synthetic_mechanism_discrimination/summarize.py
python experiments/synthetic_mechanism_discrimination/plot_figure2.py
```

`reference_results/` contains only compact, anonymized arrays/CSV values required to reproduce the submitted Figure 2 without shipping large checkpoints or synthetic datasets. To replace those compact files with a fresh completed run:

```bash
python experiments/synthetic_mechanism_discrimination/export_figure2_data.py
```
