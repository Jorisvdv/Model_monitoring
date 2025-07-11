# Model_monitoring
Monitoring code corresponding to the article [The importance of model governance in clinical AI models: case study on the relevance of data drift detection](https://bmjdigitalhealth.bmj.com/content/1/1/e000046)

## Citation:
Joris P van der Vorst, Jim M Smit, Davy van de Sande, Björn van der Ster, Freek Daams, Renske Schasfoort, Diederik Gommers, Cornelis Verhoef, Dirk J Grünhagen, Michel E van Genderen, Denise E Hilling - Importance of model governance in clinical AI models: case study on the relevance of data drift detection: BMJ Digital Health & AI 2025;1:e000046.

## Installation
Code for this project is done in Python 3.11
Required packages can be installed in two ways, using pip or uv
- `pip install requirements.txt` or `uv sync` for the essential packages for the model monitoring.
- `pip install requirements_full.txt` or `uv sync --all-extras` for all packages used in creating the Juypter notebook

## Usage
This project provides two main classes for model monitoring: `TemporalValidation` and `DataDrift`.
See [the juypter notebook](model_monitoring_report.ipynb) for an example of the use.