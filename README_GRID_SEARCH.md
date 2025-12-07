# HNSW Grid Search Tool

## Overview

This toolkit provides automated parameter grid search and analysis for HNSW index optimization.

## Files

- `grid.sh` - Main grid search script
- `grid_analysis.py` - Python analysis and visualization tool
- `grid_results/` - Output directory for results

## Usage

### 1. Run Grid Search

```bash
chmod +x grid.sh
./grid.sh
```

The script will:
- Test multiple parameter combinations (M, EFC, EFS)
- Save results to CSV with timestamp
- Display progress and summary statistics

### 2. Analyze Results

```bash
python3 grid_analysis.py
```

This will:
- Load the latest results CSV
- Display comprehensive analysis
- Generate visualization plots
- Identify Pareto-optimal configurations

## Output Files

### CSV Format
