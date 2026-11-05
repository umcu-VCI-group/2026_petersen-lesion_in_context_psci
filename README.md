## Analysis code for "Determinants of domain-specific cognitive impairment after stroke: a mega-analysis of 13 acute ischaemic stroke cohorts"

## Data layout

The code reads from and writes to a project root that holds `analysis/`, `data/` and `outputs/`. The notebooks
use the first parent folder of the working directory that contains `analysis/lib`, so run them from
`analysis/notebooks/` with `data/` placed next to `analysis/`. All results are written to `outputs/`.

```
data/tabular/clinical_data.csv                    clinical and cognitive data
data/lacuna/METAVCI_PSCI_LACUNA/derivatives/      regional lesion features (Lacuna parcelstats TSVs)
    group_ses-01_label-{acuteinfarct,WMH}_method-{rd,fnm,snm}_atlas-{ATLAS}_desc-{damagepct,tmap,disconnectionpct}_parcelstats.tsv
data/lacuna/METAVCI_PSCI_LACUNA/sub-{ID}/ses-01/anat/   per-patient maps, MNI152NLin6Asym
    *_label-acuteinfarct_space-MNI152NLin6Asym_mask.nii.gz
    *_label-acuteinfarct_method-fnm_space-MNI152NLin6Asym_desc-tmap.nii.gz
    *_label-acuteinfarct_method-snm_space-MNI152NLin6Asym_desc-disconnectionpct.nii.gz
    *_label-WMH_space-MNI152NLin6Asym_mask.nii.gz
data/lacuna/METAVCI_PSCI_CHRONIC_LACUNA/derivatives/   regional features of pre-existing lesions
data/lacuna/subsamples/METAVCI_PSCI_LACUNA_sub{10,25,50,75}pct/derivatives/   structural LNM from tractogram subsamples
outputs/feature_importance/yabplot_custom_atlases/nettekoven32/   yabplot cerebellar atlas for rendering
```

Regional features cover 190 regions: Schaefer-100 (7 networks), Tian S1 (16), Nettekoven (32) and XTRACT (42).
The parcellation sensitivity analysis (notebook 07) additionally needs the Schaefer-Tian 450 and 1054, Tian S2 (32)
and default Nettekoven parcelstats. Lesion location (`rd`), functional LNM (`fnm`, GSP1000) and structural LNM
(`snm`, dTOR985) features were derived with [Lacuna](https://github.com/m-petersen/lacuna)
([documentation](https://lacuna-py.readthedocs.io)).

The clinical data and lesion data belong to the contributing cohorts of the Meta VCI Map consortium and are not
distributed. Please refer to the respective references for information on atlases, connectomes and templates.

| step | file | output |
|---|---|---|
| 1 | `01_pooled_cv.ipynb` | pooled 10 × 10 cross-validation: baseline, single-representation and integrated models, null models, drop-one-representation ablation |
| 2 | `02_loco_cv.ipynb` | leave-one-cohort-out cross-validation |
| 3 | `03_single_cohort_cv.ipynb` | single-cohort reference (Bundang VCI) |
| 4 | `04_backend_comparison.ipynb` | XGBoost and TabPFN base learners (GPU for TabPFN) |
| 5 | `05_sample_size_curves.ipynb` | sample-size curves |
| 6 | `06_branch_preexisting.ipynb` | integrated model with pre-existing lesion features (WMH, old infarcts, lacunes, all) |
| 7 | `07_sensitivity_atlas.ipynb` | higher-resolution parcellations |
| 8 | `08_sensitivity_streamlines.ipynb` | structural LNM from tractogram subsamples |
| 9 | `09_sensitivity_regional_vs_voxelwise.ipynb` | regional vs voxelwise (PCA) lesion features, Supplementary figure S11 |
| 10 | `10_wmh_volume_analyses.ipynb` | WMH volume vs topography; models stratified by infarct volume, age and sex |
| 11 | `11_baseline_variable_ablation.ipynb` | Supplementary figure S5 |
| 12 | `12_collect_predictions.ipynb` | per-patient out-of-fold predictions |
| 13 | `13_umap_clustering.ipynb` | UMAP and clustering, cluster and lesion distribution maps, Supplementary figures S1, S19 |
| 14 | `14_ridge_beta_maps.ipynb` | ridge coefficients and brain renders |
| 15 | `15_descriptive_statistics.ipynb` | Table 1, Supplementary table S1 |
| 16 | `16_figs_performance.ipynb` | Supplementary figures S2–S4, S13, S15 |
| 17 | `17_figs_sensitivity.ipynb` | Supplementary figures S8–S10 |
| 18 | `18_figs_wmh_loco_panels.ipynb` | Supplementary figures S6, S7, S14, S22 |
| 19 | `19_compose_figures.ipynb` | Figures 2–5, Supplementary figures S12, S16–S18, S20, S21 |

## Environment

Python 3.11 with numpy, pandas, scipy, scikit-learn, joblib, matplotlib, seaborn, adjustText, nibabel, nilearn,
umap-learn, hdbscan and threadpoolctl. Notebook 04 additionally needs xgboost and tabpfn; notebook 14 needs
pyvista and yabplot (off-screen rendering via xvfb). Versions used for the paper:

```
python 3.11.10      numpy 2.3.5        pandas 2.3.3       scipy 1.15.3       scikit-learn 1.8.0
joblib 1.5.3        matplotlib 3.10.8  seaborn 0.13.2     adjustText 1.3.0   nibabel 5.3.3
nilearn 0.13.0      umap-learn 0.5.12  hdbscan 0.8.43     xgboost 3.2.0      tabpfn 6.4.1
pyvista 0.47.3      yabplot 0.4.0      lacuna 0.0.1.dev419
```

> [!TIP]
> Lesion features were computed with [Lacuna](https://github.com/m-petersen/lacuna), which is needed to regenerate `data/lacuna/` from lesion masks.

## Code structure

```
analysis/lib/data.py       parcelstats loading, patient pools, baseline design matrix, targets
analysis/lib/cv.py         pooled and leave-one-cohort-out splits, sample-size curves
analysis/lib/pipelines.py  ridge, XGBoost and TabPFN pipelines; single-model and null-model fold runners
analysis/lib/stacking.py   stacked (integrated) model fold runner
analysis/lib/stats.py      Nadeau–Bengio corrected t-test, Benjamini–Hochberg FDR
analysis/lib/plotting.py   labels, colours, significance tables and markers, figure style
analysis/lib/atlases.py    region-name mapping for brain renders
```
