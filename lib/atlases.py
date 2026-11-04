"""Atlas helpers for ROI-level GLM visualizations with yabplot.

Splits the combined Schaefer-2018-Tian-2020 1054-parcel atlas into a 1000-parcel
cortical block (Schaefer 17Networks) and a 54-parcel subcortical block (Tian S4),
and provides naming-map dicts that translate the parcel-TSV column names used by
the predictive modeling pipeline into the names expected by yabplot's built-in
or custom atlases.
"""

from __future__ import annotations

import re
from pathlib import Path


# --------------------------------------------------------------------------
# Parcel splitting: the schaefer2018tian2020 1054 atlas combines cortex + subcortex
# in a single TSV. Cortex parcel names start with "17Networks_"; subcortex names
# follow the Tian S4 naming convention (e.g. "THA-DP-lh", "NAc-shell-lh").
# --------------------------------------------------------------------------
def split_schaefer_tian_parcels(parcels):
    """Return (cortex_parcels, subcortex_parcels) lists from the combined 1054 list."""
    cortex = [p for p in parcels if p.startswith("17Networks_")]
    subcortex = [p for p in parcels if not p.startswith("17Networks_")]
    return cortex, subcortex


# --------------------------------------------------------------------------
# XTRACT — full-name → yabplot xtract_tiny abbreviation.
# --------------------------------------------------------------------------
XTRACT_MAPPING = {
    "Anterior Commissure": "AC",
    "Arcuate Fasciculus L": "AF_L",
    "Arcuate Fasciculus R": "AF_R",
    "Acoustic Radiation L": "AR_L",
    "Acoustic Radiation R": "AR_R",
    "Anterior Thalamic Radiation L": "ATR_L",
    "Anterior Thalamic Radiation R": "ATR_R",
    "Cingulum subsection: Dorsal L": "Cing_Dorsal_L",
    "Cingulum subsection: Dorsal R": "Cing_Dorsal_R",
    "Cingulum subsection: Peri-genual L": "Cing_Peri_L",
    "Cingulum subsection: Peri-genual R": "Cing_Peri_R",
    "Cingulum subsection: Temporal L": "Cing_Temp_L",
    "Cingulum subsection: Temporal R": "Cing_Temp_R",
    "Corticospinal Tract L": "CST_L",
    "Corticospinal Tract R": "CST_R",
    "Frontal Aslant Tract L": "FAT_L",
    "Frontal Aslant Tract R": "FAT_R",
    "Forceps Major": "FMaj",
    "Forceps Minor": "FMin",
    "Fornix L": "Fornix_L",
    "Fornix R": "Fornix_R",
    "Inferior Fronto-Occipital Fasciculus L": "IFOF_L",
    "Inferior Fronto-Occipital Fasciculus R": "IFOF_R",
    "Inferior Longitudinal Fasciculus L": "ILF_L",
    "Inferior Longitudinal Fasciculus R": "ILF_R",
    "Middle Cerebellar Peduncle": "MCP",
    "Middle Longitudinal Fasciculus L": "MdLF_L",
    "Middle Longitudinal Fasciculus R": "MdLF_R",
    "Optic Radiation L": "OR_L",
    "Optic Radiation R": "OR_R",
    "Superior Longitudinal Fasciculus 1 L": "SLF1_L",
    "Superior Longitudinal Fasciculus 1 R": "SLF1_R",
    "Superior Longitudinal Fasciculus 2 L": "SLF2_L",
    "Superior Longitudinal Fasciculus 2 R": "SLF2_R",
    "Superior Longitudinal Fasciculus 3 L": "SLF3_L",
    "Superior Longitudinal Fasciculus 3 R": "SLF3_R",
    "Superior Thalamic Radiation L": "STR_L",
    "Superior Thalamic Radiation R": "STR_R",
    "Uncinate Fasciculus L": "UF_L",
    "Uncinate Fasciculus R": "UF_R",
    "Vertical Occipital Fasciculus L": "VOF_L",
    "Vertical Occipital Fasciculus R": "VOF_R",
}


def xtract_data_dict(values_by_full_name):
    """Convert a {full-tract-name: value} dict into a {yabplot_abbrev: value} dict."""
    return {
        XTRACT_MAPPING[name]: v
        for name, v in values_by_full_name.items()
        if name in XTRACT_MAPPING
    }


# --------------------------------------------------------------------------
# Tian S4 — programmatic. The parcel-TSV uses names like "THA-DP-lh"; yabplot's
# Melbourne_S4 / tian_s4 atlas uses "Tha-DP-lh" or similar. The transform is
# usually a simple case fix on the first three characters.
# --------------------------------------------------------------------------
def tian_data_dict(values_by_parcel_name, custom_mapping=None):
    """Default identity mapping. Override with `custom_mapping` if needed.

    For most use cases yabplot's tian atlas accepts the same case-conventions as
    our TSV. If your `plot_subcortical` call complains about unmatched regions,
    pass a custom_mapping = {tsv_name: yabplot_name} dict.
    """
    if custom_mapping is None:
        return dict(values_by_parcel_name)
    return {
        custom_mapping.get(k, k): v
        for k, v in values_by_parcel_name.items()
    }


# --------------------------------------------------------------------------
# Nettekoven — programmatic, similar story. Parcel names like "M1La" map to
# yabplot's nettekoven names which are typically the same.
# --------------------------------------------------------------------------
def nettekoven_data_dict(values_by_parcel_name, custom_mapping=None):
    """Default identity mapping. See tian_data_dict for override semantics."""
    if custom_mapping is None:
        return dict(values_by_parcel_name)
    return {
        custom_mapping.get(k, k): v
        for k, v in values_by_parcel_name.items()
    }


# --------------------------------------------------------------------------
# Schaefer 17Networks — cortex side of the schaefer2018tian2020 1054 atlas.
# Yabplot ships schaefer1000 with 7Networks ordering by default; for 17Networks
# you need a custom atlas built via wb_command. See yabplot documentation 
# for the build recipe.
#
# The TSV parcel name format is: "17Networks_LH_VisCent_Striate_1" etc.
# A custom-built schaefer1000_17net atlas yields region names matching this
# convention exactly (one row per parcel in the LUT). The data dict is then
# {parcel: value}.
# --------------------------------------------------------------------------
def schaefer_data_dict(values_by_parcel_name, custom_mapping=None):
    """Default identity mapping; override if your custom atlas uses different keys."""
    if custom_mapping is None:
        return dict(values_by_parcel_name)
    return {
        custom_mapping.get(k, k): v
        for k, v in values_by_parcel_name.items()
    }
