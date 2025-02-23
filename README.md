# My Data Analysis - Documentation

## Overview

**Data Analysis** is a Python package designed for efficient data loading, cleaning, visualization, feature analysis, and outlier detection. It provides various utilities to preprocess and analyze datasets using **pandas, seaborn, matplotlib, and scikit-learn**.

## Installation

To install the package, run:

```bash
pip install .
```

or if hosted on a repository:

```bash
pip install git+https://github.com/nelson960/Data_Analysis-Tool.git
```

## Usage Guide

### Importing the Package

```python
from my_data_tool import DataLoader, DataCleaner, DataVisualizer, DataReporter, FeatureAnalyzer, OutlierDetector
```

### 1. Data Loading

```python
loader = DataLoader("data.csv")
df = loader.get_dataframe()
selected_df = loader.select_columns(["column1", "column2"])
```

Supports CSV, JSON, and Parquet formats.

### 2. Data Cleaning

```python
cleaner = DataCleaner(df)
df_cleaned = cleaner.drop_columns(["unnecessary_column"])
df_cleaned = cleaner.fill_missing_values(0)
df_cleaned = cleaner.clean_column("text_column")
cleaner.show_changes_log()
```

### 3. Data Visualization

```python
visualizer = DataVisualizer(df)
visualizer.visualize_missing_values()
visualizer.correlation_heatmap()
visualizer.visualize_data_distribution()
```

### 4. Data Reporting

```python
reporter = DataReporter(df, visualizer)
reporter.generate_report()
```

### 5. Feature Analysis

```python
analyzer = FeatureAnalyzer(df)
analyzer.feature_importance("target_column")
```

### 6. Outlier Detection

```python
detector = OutlierDetector(df)
outliers = detector.detect_outliers(method='zscore', threshold=3.0)
```

## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- scipy
- scikit-learn

## Author

Nelson Alex

## License

MIT License

