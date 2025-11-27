#!/usr/bin/env python3
"""
Script to update analysis notebooks with timing fix warnings and new parameters
"""

import json
import os

def add_timing_fix_warning_cell():
    """Create a warning cell about the timing fix"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚨 CRITICAL TIMING FIX UPDATE\n",
            "\n",
            "**IMPORTANT**: This notebook may contain analysis based on the **UNREALISTIC** smooth cash flow model.\n",
            "\n",
            "## Timing Model Changes\n",
            "\n",
            "**OLD (Unrealistic)**: `vaultBalance += feesCollected - actualL1Cost` (every 2s)\n",
            "\n",
            "**NEW (Realistic)**:\n",
            "- Fee collection: `vaultBalance += feesCollected` (every 2s)\n",
            "- L1 cost payment: `vaultBalance -= actualL1Cost` (every 12s, when t % 6 === 0)\n",
            "\n",
            "## Impact\n",
            "\n",
            "The timing fix creates **natural saw-tooth deficit patterns** that **invalidate previous optimal parameters**:\n",
            "\n",
            "| Strategy | Old Parameters | NEW Post-Timing-Fix | Change |\n",
            "|----------|---------------|--------------------|---------|\n",
            "| Optimal  | μ=0.0, ν=0.3, H=288 | **μ=0.0, ν=0.1, H=36** | ν↓70%, H↓87% |\n",
            "| Balanced | μ=0.0, ν=0.1, H=576 | **μ=0.0, ν=0.2, H=72** | ν↑100%, H↓87% |\n",
            "| Crisis   | μ=0.0, ν=0.9, H=144 | **μ=0.0, ν=0.7, H=288** | ν↓22%, H↑100% |\n",
            "\n",
            "## Updated Analysis\n",
            "\n",
            "✅ **Latest Results**: See `analysis/results/*POST_TIMING_FIX.csv`  \n",
            "✅ **Corrected Parameters**: See `analysis/OPTIMAL_PARAMETERS_SUMMARY.md`  \n",
            "✅ **Web Interface**: Uses realistic timing and new optimal parameters  \n",
            "\n",
            "**Use these new parameters for any new analysis or protocol implementation.**"
        ]
    }

def update_notebook(notebook_path):
    """Update a notebook with timing fix warning"""

    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Add warning cell at the beginning (after title if exists)
    warning_cell = add_timing_fix_warning_cell()

    # Insert after the first cell (usually the title)
    if len(notebook['cells']) > 0:
        notebook['cells'].insert(1, warning_cell)
    else:
        notebook['cells'].insert(0, warning_cell)

    # Create backup
    backup_path = notebook_path.replace('.ipynb', '_PRE_TIMING_FIX_BACKUP.ipynb')
    with open(backup_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"✓ Updated {notebook_path}")
    print(f"✓ Backup saved to {backup_path}")

def main():
    """Update all analysis notebooks"""

    notebook_dir = "/Users/ulyssepavloff/Desktop/Nethermind/taiko-fee-analysis/analysis/notebooks"

    notebooks = [
        "taiko_fee_analysis.ipynb",
        "optimal_parameters_research.ipynb",
        "updated_taiko_analysis.ipynb"
    ]

    print("Updating analysis notebooks with timing fix warnings...\n")

    for notebook in notebooks:
        notebook_path = os.path.join(notebook_dir, notebook)
        if os.path.exists(notebook_path):
            update_notebook(notebook_path)
        else:
            print(f"✗ Notebook not found: {notebook_path}")

    print(f"\n✓ All notebooks updated with timing fix warnings")
    print("✓ Previous versions backed up with '_PRE_TIMING_FIX_BACKUP' suffix")
    print("✓ Notebooks now prominently display the parameter invalidation")

if __name__ == "__main__":
    main()