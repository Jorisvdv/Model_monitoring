import nannyml as nml
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


class DataDrift:
    def __init__(self, timestamp_column: str, chunk_period: str = "M"):
        self.timestamp_column = timestamp_column
        self.chunk_period = chunk_period
        self.univariate_results = None
        self.multivariate_results = None

    def run_univariate_drift(
        self, reference_data: pd.DataFrame, analysis_data: pd.DataFrame, column_names: list, categorical_features: list
    ):
        uni_calc = nml.UnivariateDriftCalculator(
            column_names=column_names,
            treat_as_categorical=categorical_features,
            timestamp_column_name=self.timestamp_column,
            continuous_methods=["kolmogorov_smirnov"],
            categorical_methods=["jensen_shannon"],
            chunk_period=self.chunk_period,
        )
        uni_calc.fit(reference_data)
        self.univariate_results = uni_calc.calculate(analysis_data)
        return self.univariate_results

    def plot_univariate_drift(self, kind="drift", **kwargs):
        if self.univariate_results is None:
            print("Please run run_univariate_drift() first.")
            return None
        return self.univariate_results.plot(kind=kind, **kwargs)

    def get_univariate_alerts(self):
        if self.univariate_results is None:
            print("Please run run_univariate_drift() first.")
            return []

        results_df = self.univariate_results.to_df()
        analysis_results_df = results_df[results_df[("chunk", "period")] == "analysis"]

        alert_columns = [
            col for col in analysis_results_df.columns if isinstance(col, tuple) and len(col) > 2 and col[2] == "alert"
        ]

        if not alert_columns:
            return []

        alerts = analysis_results_df[alert_columns]
        alert_active = alerts.any()
        alert_column_names = alert_active[alert_active].index.get_level_values(0).unique().tolist()
        return alert_column_names

    def plot_univariate_distributions_for_alerts(self, **kwargs):
        if self.univariate_results is None:
            print("Please run run_univariate_drift() first.")
            return

        alert_column_names = self.get_univariate_alerts()

        if not alert_column_names:
            print("No univariate drift alerts to plot.")
            return

        print(f"Plotting distributions for columns with alerts: {', '.join(alert_column_names)}")
        for column_name in alert_column_names:
            self.univariate_results.filter(column_names=[column_name]).plot(kind="distribution", **kwargs).show()

    def run_multivariate_drift(self, reference_data: pd.DataFrame, analysis_data: pd.DataFrame, column_names: list):
        multi_calc = nml.DataReconstructionDriftCalculator(
            column_names=column_names,
            timestamp_column_name=self.timestamp_column,
            chunk_period=self.chunk_period,
            imputer_categorical=SimpleImputer(strategy="most_frequent", missing_values=np.nan),
        )
        multi_calc.fit(reference_data)
        self.multivariate_results = multi_calc.calculate(analysis_data)
        return self.multivariate_results

    def plot_multivariate_drift(self, **kwargs):
        if self.multivariate_results is None:
            print("Please run run_multivariate_drift() first.")
            return None
        return self.multivariate_results.plot(**kwargs)

    def get_multivariate_alerts(self):
        if self.multivariate_results is None:
            print("Please run run_multivariate_drift() first.")
            return False

        results_df = self.multivariate_results.to_df()
        analysis_results_df = results_df[results_df[("chunk", "period")] == "analysis"]
        alerts = analysis_results_df[("reconstruction_error", "alert")]
        return alerts.any()
