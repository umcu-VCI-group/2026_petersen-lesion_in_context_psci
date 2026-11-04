"""Predictive-modeling library shared by all compute notebooks.

Modules:
    data       — atlas TSV loading, mask-presence indices, restricted pools
    pipelines  — sklearn pipeline factories (Ridge baseline, demographics-only, HPC variants)
    stacking   — paired view-config stacker (process_stacking_fold)
    cv         — pooled and LOCO splitters, paired-fold runner
    stats      — paired permutation test, BH-FDR
"""
