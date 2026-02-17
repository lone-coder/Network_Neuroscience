#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 1: Demographic Comparisons
=================================

STUDY: Vanier et al. - "Diffuse effects of preterm birth on the white matter
       connectome of early school-aged children"

PURPOSE: Generate publication-ready Table 1 showing demographic comparisons
         between full-term (FT), moderate-to-late preterm (MLPT), and very
         preterm (VPT) groups.

OUTPUT: Table_1_Demographics.png in outputs/tables/main/

STATISTICS:
    - Continuous variables (Age, GA, Motion): Mann-Whitney U tests
    - Categorical variables (Sex, Handedness, Sibling Status): Chi-squared tests
    - All p-values corrected with Holm-Bonferroni method
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = '[PATH_TO_DATA]'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/tables/main')

# ==============================================================================
# TABLE GENERATION
# ==============================================================================

def generate_table():
    """Generate publication-ready demographic comparison table."""
    
    # Demographic data with statistical test results
    data = [
        ["", "Full-term\\n(FT)", "Moderate-to-late\\npreterm (MLPT)", "Very preterm\\n(VPT)", "FT vs\\nMLPT", "FT vs\\nVPT", "MLPT vs\\nVPT"],
        ["N (male/female)", "133 (58/75)", "34 (18/16)", "37 (16/21)", "p = 1.00", "p = 1.00", "p = 1.00"],
        ["Age (years)", "5.83 ± 0.85", "5.52 ± 1.02", "6.04 ± 1.15", "p = 0.10", "p = 0.46", "p = 0.12"],
        ["Gestational Age (weeks)", "39.14 ± 1.03", "34.47 ± 1.44", "27.70 ± 2.47", "p < 0.001", "p < 0.001", "p < 0.001"],
        ["Head Motion (# volumes excl.)", "0.32 ± 1.02", "0.55 ± 1.37", "0.63 ± 1.25", "p = 0.22", "p = 0.00", "p = 0.36"],
        ["Handedness", "", "", "", "p = 0.61", "p = 0.01", "p = 0.61"],
        ["      Left", "9", "4", "8", "", "", ""],
        ["      Right", "122", "29", "26", "", "", ""],
        ["      Ambidextrous", "2", "1", "3", "", "", ""],
        ["Sibling Status", "", "", "", "", "", ""],
        ["      Singletons", "109", "22", "27", "", "", ""],
        ["      Singleton Siblings", "24", "2", "0", "p = 0.28", "p < 0.001", "p = 0.44"],
        ["      Twins", "0", "10", "10", "p < 0.001", "p < 0.001", "p = 1.00"]
    ]

    # Font settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')

    # Create table with adjusted column widths
    col_widths = [0.22, 0.13, 0.15, 0.13, 0.12, 0.12, 0.12]
    table = ax.table(cellText=data, 
                     loc='center', 
                     cellLoc='left',
                     colWidths=col_widths)
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('none')
        
        # Header row styling
        if row == 0:
            cell.set_text_props(weight='bold', wrap=True)
            cell.set_facecolor('#F8F9FA')
        
        # Group header styling
        if row in [5, 9]:
            cell.set_text_props(weight='bold')
            
        # P-value column styling
        if col >= 4:
            cell.set_text_props(alpha=0.7)
            
    # Professional horizontal lines
    ax.axhline(y=0.92, xmin=0.02, xmax=0.98, color='black', linewidth=2.0)
    ax.axhline(y=0.82, xmin=0.02, xmax=0.98, color='black', linewidth=1.2)
    ax.axhline(y=0.08, xmin=0.02, xmax=0.98, color='black', linewidth=2.0)

    # Title
    plt.text(0.5, 0.96, 
             "Table 1. Demographic comparisons between full-term (FT), moderate-to-late preterm (MLPT),\\nand very preterm (VPT) born children.", 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=16, fontweight='bold', transform=ax.transAxes)

    # Footnote
    footnote_text = (
        "Values displayed are means ± standard deviation or counts. For continuous variables (Age, GA, Motion), pairwise comparisons\\n"
        "use Mann-Whitney U tests. For categorical variables (Sex, Handedness, Sibling Status), pairwise comparisons use Chi-squared tests.\\n"
        "All p-values are corrected for multiple comparisons (Holm-Bonferroni)."
    )
    plt.text(0.02, 0.02, footnote_text, 
             horizontalalignment='left', verticalalignment='bottom', 
             fontsize=10, style='italic', transform=ax.transAxes)

    # Save table
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'Table_1_Demographics.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    print(f"✓ Table 1 saved to: {output_path}")

if __name__ == "__main__":
    generate_table()
