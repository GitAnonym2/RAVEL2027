# RAVEL

**RAVEL: Factorizing Multi-Region Neural Communication into Reusable Propagation Routes and Dynamic Recruitment**

Anonymous code release accompanying an ICLR 2027 submission. This repository contains the reusable RAVEL implementation, unit tests, dataset-neutral examples, and the synthetic mechanism-discrimination experiment used in Figure 2.

RAVEL is a directed conditional spectral model. For each ordered source-target pair it separates (1) a small dictionary of delayed, frequency-dependent propagation atoms from (2) event-time-varying nonnegative recruitment. Recording-specific communication coordinates, spatial residue matrices, nuisance coupling, and recruitment are not pooled across recordings.

## Quick start

Python 3.11 is the reference environment used for the submitted experiments.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python examples/quickstart_synthetic.py
```

The core package itself requires NumPy, pandas, and SciPy. `requirements.txt` pins the main versions used in the reference environment and adds Matplotlib for figure reproduction. The complete environment snapshot used during development is archived under `environment/requirements_full_reference.txt`; it is intentionally not the default installation route because it contains platform/CUDA-specific packages.

## Applying RAVEL to your own recordings

The default paper configuration expects aligned source and target epochs with shape `[trial, time, channel]`, sampled at 500 Hz over -1 to +2 s. Use `ravel.build_route_session_from_epochs(...)` so that preprocessing, train/validation/test splitting, communication-coordinate estimation, multitaper features, and target evaluation remain consistent. See `examples/dataset_adapter_template.py`.

A minimal workflow is:

```python
from ravel import build_route_session_from_epochs, fit_independent_session, paper_config

session = build_route_session_from_epochs(
    source_epochs=source_epochs,
    target_epochs=target_epochs,
    metadata=metadata,
    route_name="SOURCE_to_TARGET",
    source_region="SOURCE",
    target_region="TARGET",
    session_id="session_001",
    subject_id="subject_001",
    d=3,
    reference_context="active",
    config=paper_config(),
)

fit = fit_independent_session(
    session, motif_count=2, context="active",
    checkpoint_dir="checkpoints/session_001",
    config=paper_config(),
)
print([(a.delay_ms, a.pole_hz) for a in fit["atoms"]])
```

## Repository map

- `ravel/`: reusable implementation of the conditional spectral model, preprocessing, optimization, metrics, pooling, and transfer.
- `examples/quickstart_synthetic.py`: small executable end-to-end example.
- `examples/dataset_adapter_template.py`: template for real aligned-epoch datasets.
- `experiments/synthetic_mechanism_discrimination/`: Figure 2 simulation, matched dynamic-delay comparator, optional ADM wrapper, summary, and figure reproduction.
- `tests/`: seven unit/regression tests for preprocessing, optimizer behavior, I/O, and the frozen numerical fixtures.
- `configs/ravel_default.json`: machine-readable default paper configuration.
- `environment/requirements_full_reference.txt`: archival full environment snapshot.

## Baselines and third-party code

Third-party model source trees should generally **not** be copied into this repository. For ADM, MRM-GP, and DLAG, the reproducible release should instead include the exact wrapper/evaluation scripts used in the paper, the upstream repository/version or commit, any required patch notes, and the shared train/test split and evaluator. This avoids relicensing other authors' code while still allowing the paper comparison to be reproduced.

The synthetic folder already follows that pattern for ADM: `run_adm.py` expects a separate ADM checkout. Wrappers for the Allen Visual Coding baselines and the corresponding Allen data preparation scripts should be added alongside the real-data experiments before the final anonymous release is considered complete.

## Anonymity and generated outputs

This review bundle contains no author names, affiliations, e-mail addresses, local home-directory paths, or lab identifiers. Large generated datasets/checkpoints are excluded by `.gitignore`; scripts regenerate them from the fixed seeds. Compact reference arrays needed for Figure 2 plotting are included.

## License

A software license is intentionally not assigned in this anonymous review bundle. Add the intended open-source license and author metadata for the archival public release after double-blind review.
