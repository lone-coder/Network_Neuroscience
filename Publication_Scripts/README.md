# Vanier Preterm Connectome Analysis Pipeline

> **Study**: Diffuse effects of preterm birth on the white matter connectome of early school-aged children

This folder contains the consolidated, cleaned analysis scripts organized by pipeline stage.

---

## Pipeline Order

Run scripts in the following order:

### 1. Preprocessing (`01_preprocessing/`)
- **`Creating_cross_sectional_data.py`**: Prepares subject data for analysis
  - Input: Raw demographic and imaging data
  - Output: Cross-sectional data files organized by group

- **`generate_demographics_table.py`**: Creates Table 1 with demographic comparisons
  - Input: Hard-coded demographic statistics
  - Output: `Table_1_Demographics.png` in `/outputs/tables/main/`
  - Statistics: Mann-Whitney U (continuous) and Chi-squared (categorical) with Holm correction

### 2. Linear Mixed Models (`02_linear_mixed_models/`)
- **`Aug_2025_LMM.R`**: Runs edge-level LMM analysis
  - Input: Weighted connectome matrices, demographic data
  - Output: `Combined_All_Subjects_LMM_Results.csv` with standardized β coefficients
  - Model: `Edge ~ GA + Age + Gender + Handedness + Motion + (1|Subject)`

### 3. Edge Processing (`03_edge_processing/`)
- **`edge_labeling_and_categorization.py`**: Labels edges and categorizes by distance/RC
  - Input: LMM results, atlas labels, group-averaged matrices
  - Output: 
    - `edge_labels_*.csv` (anatomical labels)
    - `distance_categorization.csv` (Short/Medium/Long)
    - `*_RC_FILTERED.csv` (Rich-club/Feeder/Local)

### 4. Rich-Club Analysis (`05_rich_club/`)
- **`rich_club_curves.py`**: Processes RC coefficient curves and performs group comparisons  (Figure 2, Tables 2/S1/S2)
  - Input: Individual RC pickle files organized by group
  - Output: Mean RC curves with CIs, statistical comparison tables

- **`calculate_individual_overlap.py`**: Computes rich-club consistency (Figure S3)
  - Input: Group-averaged and individual connectivity matrices
  - Output: Overlap statistics and publication figure

### 5. Statistical Analysis (`04_statistical_analysis/`)
- **`lmm_category_statistics.py`**: Edge β comparisons by category (Tables 3-4, S3-S4)
  - Tests: Kruskal-Wallis + Mann-Whitney U with Holm correction
  - Distance: n=6 corrections (3 pairs × 2 directions)
  - RC: n=18 corrections (3 pairs × 2 directions × 3 levels)

- **`network_density_analysis.py`**: Global network density comparisons (Figure 4)
  - Uses combined thresholding (distance > 0 AND weight > 0)
  - Tests: Kruskal-Wallis omnibus + Mann-Whitney U pairwise (Holm-corrected)
  - Output: `unweighted_density_thresholded.png`

- **`network_based_statistics.py`**: NBS supplementary validation (Figure S1 components)
  - Purpose: Identify connected edge components showing group differences
  - Threshold: Cohen's d = 0.1 → t = 0.76 (harmonic mean conversion)
  - Output: Component CSV files for downstream heatmap visualization
  - Note: Supplementary validation of LMM findings


### 6. Visualization (`06_visualization/`)
- **`heatmap.py`**: Generates β coefficient heatmaps (Figures 1, S1, S2)
  - Input: LMM edge results (`edge_labels_*.csv`)
  - Output: Heatmaps for gestational age (Fig 1), NBS filtered (Fig S1), age at scan (Fig S2)

- **`create_figure3.py`**: Violin plots by edge category (Figure 3)
  - Shows positive and negative β distributions for RC and distance categories

- **`create_figure_s4.py`**: Top 20 most frequent rich-club nodes (Figure S4)
- **`create_figure_s5.py`**: Edge categorization bar charts (Figure S5)
- **`create_figure_s6.py`**: Violin plots for 15% and 20% RC thresholds (Figure S6)

---

## Key Methodological Decisions

| Decision | Implementation | Script Location |
|----------|----------------|-----------------|
| Edge filtering | BOTH distance > 0 AND weight > 0 | `03_edge_processing/` |
| RC statistics | Single-percentile (one value/subject) | `04_rich_club_analysis/` |
| Multiple correction | Holm-Bonferroni | All statistical scripts |
| Per-edge standardization | Outcomes standardized per-edge | `02_linear_mixed_models/` |

---

## Dependencies

### R
```r
lme4, lmerTest, dplyr, readr, parallel
```

### Python
```python
numpy, pandas, scipy, matplotlib, seaborn, networkx, bctpy
```

---

## Deprecated Scripts (NOT included)

The following older versions should NOT be used:
- `Dec_2025 _network_density.py` → Uses Bonferroni instead of Holm
- `Updated LMM processing.py` → Uses distance-only filtering
- `updatedPickles_RCC.py` → Uses aggregated percentiles (pseudo-replication)
- `July_2025_latest.R` → Older LMM implementation
