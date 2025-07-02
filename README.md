# Model_monitoring
Monitoring code corresponding to the article [The importance of model governance in clinical AI models: case study on the relevance of data drift detection](http://dx.doi.org/10.1136/bmjdhai-2025-000046)

## Citation:

The importance of model governance in clinical AI models: case study on the relevance of data drift detection
Joris van der Vorst, Jim Smit, Davy van de Sande, Björn van der Ster, Freek Daams, Renske Schasfoort, Diederik Gommers, Cornelis Verhoef, Dirk Grünhagen, Michel van Genderen, Denise Hilling  
*BMJ Digital Health & AI 2025;1:e000046.*
(http://dx.doi.org/10.1136/bmjdhai-2025-000046)

## Installation


## Usage

This project provides two main classes for model monitoring: `TemporalValidation` and `DataDrift`.

### TemporalValidation

The `TemporalValidation` class in `results.py` can be used to calculate and plot monthly performance metrics for your models. Here's an example of how to use it:

```python
from results import TemporalValidation
import pandas as pd

# Assuming you have a DataFrame `test_data` with your test data
# and a dictionary `models` with your trained models

temporal_validation = TemporalValidation(
    models=models,
    target_column="your_target_column",
    date_column="your_date_column",
)
performance_df = temporal_validation.calculate_monthly_performance(test_data)
fig = temporal_validation.plot_monthly_performance(
    title="Model Performance Over Time",
    auc_lower_limit=0.8,
    brier_upper_limit=0.2,
)
if fig:
    fig.show()
```

### DataDrift

The `DataDrift` class in `data_monitoring.py` can be used to detect data drift between a reference and analysis dataset. Here's an example of how to use it:

```python
from data_monitoring import DataDrift
import pandas as pd

# Assuming you have a reference DataFrame `X_train` and an analysis DataFrame `X_test`

drift_monitor = DataDrift(
    timestamp_column="your_timestamp_column",
)
drift_monitor.run_univariate_drift(
    reference_data=X_train,
    analysis_data=X_test,
    column_names=X_train.columns.tolist(),
    categorical_features=your_categorical_features,
)

univariate_alerts = drift_monitor.get_univariate_alerts()
if univariate_alerts:
    print(f"Univariate drift detected in columns: {univariate_alerts}")
    drift_monitor.plot_univariate_distributions_for_alerts()

```
