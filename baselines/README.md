# Baseline integration

Do not vendor third-party baseline repositories here unless their license explicitly permits redistribution and there is a strong reason to do so. The preferred layout is one thin wrapper per baseline plus a README recording the upstream URL/commit, environment, data interface, native fitting objective, model-selection rule, and conversion to the shared complete-target evaluator.

For the submitted paper, add the exact wrappers used for MRM-GP, DLAG, and the Allen Visual Coding ADM analysis here (or under the corresponding experiment folder). The synthetic ADM wrapper is already under `experiments/synthetic_mechanism_discrimination/run_adm.py`.
