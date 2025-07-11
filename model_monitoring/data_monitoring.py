"""Enhanced Data Drift Monitoring Class"""

import logging
from typing import Any, Dict, List, Optional

import nannyml as nml
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    """
    A comprehensive data drift monitoring class that provides univariate and multivariate drift detection
    capabilities using NannyML.

    This class provides methods to:
    - Run univariate drift detection on individual features
    - Run multivariate drift detection across all features
    - Identify periods and columns with drift alerts
    - Generate visualizations for drift analysis
    - Export results to Excel for reporting

    Parameters
    ----------
    timestamp_column : str
        Name of the column containing timestamps
    chunk_period : str, default="M"
        Time period for chunking data (e.g., "M" for monthly, "W" for weekly)
    categorical_features : List[str], optional
        List of categorical feature names
    continuous_methods : List[str], default=["kolmogorov_smirnov"]
        Methods to use for continuous features drift detection
    categorical_methods : List[str], default=["jensen_shannon"]
        Methods to use for categorical features drift detection
    """

    def __init__(
        self,
        timestamp_column: str,
        chunk_period: str = "M",
        categorical_features: Optional[List[str]] = None,
        continuous_methods: Optional[List[str]] = None,
        categorical_methods: Optional[List[str]] = None,
    ):
        self.timestamp_column = timestamp_column
        self.chunk_period = chunk_period
        self.categorical_features = categorical_features or []
        self.continuous_methods = continuous_methods or ["kolmogorov_smirnov"]
        self.categorical_methods = categorical_methods or ["jensen_shannon"]

        # Data storage
        self.reference_data: Optional[pd.DataFrame] = None
        self.analysis_data: Optional[pd.DataFrame] = None

        # Results cache
        self.univariate_results: Optional[nml.base.AbstractResult] = None
        self.multivariate_results: Optional[nml.base.AbstractResult] = None

        # Track if results are outdated
        self._univariate_outdated = True
        self._multivariate_outdated = True

    def set_reference_data(self, data: pd.DataFrame) -> None:
        """
        Set the reference dataset for drift detection.

        Parameters
        ----------
        data : pd.DataFrame
            Reference dataset containing features and timestamp column

        Raises
        ------
        ValueError
            If timestamp column is not found in the data
        """
        if self.timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found in reference data")

        self.reference_data = data.copy()
        self._invalidate_results()
        logger.info(f"Reference data set with {len(data)} rows")

    def set_analysis_data(self, data: pd.DataFrame, append: bool = False) -> None:
        """
        Set or append to the analysis dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Analysis dataset containing features and timestamp column
        append : bool, default=False
            If True, append to existing analysis data. If False, replace it.

        Raises
        ------
        ValueError
            If timestamp column is not found in the data
        """
        if self.timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found in analysis data")

        if append and self.analysis_data is not None:
            self.analysis_data = pd.concat([self.analysis_data, data], ignore_index=True)
            logger.info(f"Appended {len(data)} rows to analysis data")
        else:
            self.analysis_data = data.copy()
            logger.info(f"Analysis data set with {len(data)} rows")

        self._invalidate_results()

    def run_univariate_drift(
        self, column_names: Optional[List[str]] = None, force_recalculate: bool = False
    ) -> nml.base.AbstractResult:
        """
        Run univariate drift detection on specified columns.

        Parameters
        ----------
        column_names : List[str], optional
            List of column names to analyze. If None, analyzes all columns except timestamp
        force_recalculate : bool, default=False
            If True, recalculate even if cached results exist

        Returns
        -------
        nml.base.AbstractResult
            Univariate drift detection results

        Raises
        ------
        ValueError
            If reference or analysis data is not set
        """
        self._validate_data_set()

        if not force_recalculate and not self._univariate_outdated and self.univariate_results is not None:
            logger.info("Using cached univariate results")
            return self.univariate_results

        if column_names is None:
            column_names = [col for col in self.reference_data.columns if col != self.timestamp_column]

        logger.info(f"Running univariate drift detection on {len(column_names)} columns")

        uni_calc = nml.UnivariateDriftCalculator(
            column_names=column_names,
            treat_as_categorical=self.categorical_features,
            timestamp_column_name=self.timestamp_column,
            continuous_methods=self.continuous_methods,
            categorical_methods=self.categorical_methods,
            chunk_period=self.chunk_period,
        )

        uni_calc.fit(self.reference_data)
        self.univariate_results = uni_calc.calculate(self.analysis_data)
        self._univariate_outdated = False

        return self.univariate_results

    def get_univariate_alerts(self) -> List[str]:
        """
        Get list of column names that have univariate drift alerts.

        Returns
        -------
        List[str]
            List of column names with drift alerts
        """
        if self.univariate_results is None:
            logger.warning("No univariate results available. Run run_univariate_drift() first.")
            return []

        results_df = self.univariate_results.to_df()
        analysis_period = results_df[("chunk", "chunk", "period")] == "analysis"
        alert_slice = pd.IndexSlice[:, :, "alert"]

        try:
            alert_active = results_df.loc[analysis_period, alert_slice].any()
            if not alert_active.any():
                return []

            alert_column_names = alert_active[alert_active].index.get_level_values(0).unique().tolist()
            return alert_column_names
        except Exception as e:
            logger.error(f"Error getting univariate alerts: {e}")
            return []

    def get_univariate_alert_periods(self) -> List[str]:
        """
        Get list of time periods that have univariate drift alerts.

        Returns
        -------
        List[str]
            List of time periods with drift alerts
        """
        if self.univariate_results is None:
            logger.warning("No univariate results available. Run run_univariate_drift() first.")
            return []

        results_df = self.univariate_results.to_df()
        analysis_period = results_df[("chunk", "chunk", "period")] == "analysis"
        alert_slice = pd.IndexSlice[:, :, "alert"]

        try:
            alert_rows = results_df.loc[analysis_period, alert_slice].any(axis=1)
            alert_periods = results_df.loc[alert_rows[alert_rows].index, ("chunk", "chunk", "key")].unique().tolist()
            return alert_periods
        except Exception as e:
            logger.error(f"Error getting univariate alert periods: {e}")
            return []

    def plot_univariate_drift(
        self,
        column_names: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        kind: str = "drift",
        **kwargs,
    ) -> Any:
        """
        Plot univariate drift results.

        Parameters
        ----------
        column_names : List[str], optional
            Specific columns to plot. If None, plots all columns
        methods : List[str], optional
            Specific methods to plot. If None, plots all methods
        kind : str, default="drift"
            Type of plot ("drift" or "distribution")
        **kwargs
            Additional arguments passed to the plot function

        Returns
        -------
        plotly.graph_objects.Figure
            The generated plot figure
        """
        if self.univariate_results is None:
            logger.error("No univariate results available. Run run_univariate_drift() first.")
            return None

        filtered_results = self.univariate_results

        if column_names is not None:
            filtered_results = filtered_results.filter(column_names=column_names)

        if methods is not None:
            filtered_results = filtered_results.filter(methods=methods)

        return filtered_results.plot(kind=kind, **kwargs)

    def plot_univariate_distributions_for_alerts(self, **kwargs) -> None:
        """
        Plot distributions for all columns that have drift alerts.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the plot function
        """
        alert_column_names = self.get_univariate_alerts()

        if not alert_column_names:
            logger.info("No univariate drift alerts to plot.")
            return

        logger.info(f"Plotting distributions for columns with alerts: {', '.join(alert_column_names)}")

        for column_name in alert_column_names:
            figure = self.univariate_results.filter(column_names=[column_name]).plot(kind="distribution", **kwargs)
            figure.show()

    def run_multivariate_drift(
        self,
        column_names: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None,
        force_recalculate: bool = False,
    ) -> nml.base.AbstractResult:
        """
        Run multivariate drift detection using data reconstruction with PCA.

        Parameters
        ----------
        column_names : List[str], optional
            Specific columns to include. If None, uses all columns except timestamp
        exclude_columns : List[str], optional
            Columns to exclude from analysis
        force_recalculate : bool, default=False
            If True, recalculate even if cached results exist

        Returns
        -------
        nml.base.AbstractResult
            Multivariate drift detection results

        Raises
        ------
        ValueError
            If reference and analysis data don't have matching columns
        TypeError
            If reference and analysis data don't have matching dtypes
        """
        self._validate_data_set()

        if not force_recalculate and not self._multivariate_outdated and self.multivariate_results is not None:
            logger.info("Using cached multivariate results")
            return self.multivariate_results

        # Prepare data copies for multivariate analysis
        ref_data = self.reference_data.copy()
        analysis_data = self.analysis_data.copy()

        # Validate data compatibility
        if set(ref_data.columns) != set(analysis_data.columns):
            raise ValueError("Reference and analysis data must have the same columns.")

        if not ref_data.dtypes.equals(analysis_data.dtypes):
            raise TypeError("Reference and analysis data must have the same dtypes.")

        # Determine columns to use
        if column_names is None:
            column_names = [col for col in ref_data.columns if col != self.timestamp_column]

        if exclude_columns:
            column_names = [col for col in column_names if col not in exclude_columns]

        # Prepare data for multivariate analysis
        ref_data, analysis_data = self._prepare_data_for_multivariate(ref_data, analysis_data, column_names)

        logger.info(f"Running multivariate drift detection on {len(column_names)} columns")

        multi_calc = nml.DataReconstructionDriftCalculator(
            column_names=column_names,
            timestamp_column_name=self.timestamp_column,
            chunk_period=self.chunk_period,
            imputer_categorical=SimpleImputer(strategy="most_frequent", missing_values=np.nan),
        )

        multi_calc.fit(ref_data)
        self.multivariate_results = multi_calc.calculate(analysis_data)
        self._multivariate_outdated = False

        return self.multivariate_results

    def get_multivariate_alerts(self) -> pd.Series:
        """
        Get periods with multivariate drift alerts.

        Returns
        -------
        pd.Series
            Boolean series indicating which periods have multivariate drift alerts
        """
        if self.multivariate_results is None:
            logger.warning("No multivariate results available. Run run_multivariate_drift() first.")
            return pd.Series(dtype=bool)

        results_df = self.multivariate_results.to_df()
        analysis_results_df = results_df[results_df[("chunk", "period")] == "analysis"]
        alert_mask = analysis_results_df[("reconstruction_error", "alert")]
        alerts = analysis_results_df.loc[alert_mask, ("chunk", "key")]
        return alerts.to_list()

    def plot_multivariate_drift(self, **kwargs) -> Any:
        """
        Plot multivariate drift results.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to the plot function

        Returns
        -------
        plotly.graph_objects.Figure
            The generated plot figure
        """
        if self.multivariate_results is None:
            logger.error("No multivariate results available. Run run_multivariate_drift() first.")
            return None

        return self.multivariate_results.plot(**kwargs)

    def generate_drift_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of drift detection results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing summary statistics and alert information
        """
        summary = {
            "univariate": {
                "alerts_detected": len(self.get_univariate_alerts()),
                "alert_columns": self.get_univariate_alerts(),
                "alert_periods": self.get_univariate_alert_periods(),
            },
            "multivariate": {
                "alerts_detected": 0,
                "alert_periods": [],
            },
        }

        # Add multivariate summary if available
        multivariate_alerts = self.get_multivariate_alerts()
        if len(multivariate_alerts) > 0:
            summary["multivariate"]["alerts_detected"] = len(multivariate_alerts)
            summary["multivariate"]["alert_periods"] = multivariate_alerts

        return summary

    def export_results_to_excel(self, filepath: str, include_multivariate: bool = True) -> None:
        """
        Export drift detection results to Excel with separate sheets.

        Parameters
        ----------
        filepath : str
            Path to save the Excel file
        include_multivariate : bool, default=True
            Whether to include multivariate results
        """
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Export univariate results
            if self.univariate_results is not None:
                uni_df = self.univariate_results.to_df()
                uni_df.to_excel(writer, sheet_name="Univariate_Drift")
                logger.info("Exported univariate results to Excel")

            # Export multivariate results
            if include_multivariate and self.multivariate_results is not None:
                multi_df = self.multivariate_results.to_df()
                multi_df.to_excel(writer, sheet_name="Multivariate_Drift")
                logger.info("Exported multivariate results to Excel")

            # Export summary
            summary_df = pd.DataFrame([self.generate_drift_summary()])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            logger.info(f"Results exported to {filepath}")

    def _prepare_data_for_multivariate(
        self, ref_data: pd.DataFrame, analysis_data: pd.DataFrame, column_names: List[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for multivariate analysis by handling categorical and boolean columns.

        Parameters
        ----------
        ref_data : pd.DataFrame
            Reference data
        analysis_data : pd.DataFrame
            Analysis data
        column_names : List[str]
            Column names to prepare

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Prepared reference and analysis data
        """
        # Handle categorical columns
        for col in column_names:
            if isinstance(ref_data[col].dtype, pd.CategoricalDtype):
                ref_data[col] = ref_data[col].cat.codes
                analysis_data[col] = analysis_data[col].cat.codes

        # Handle boolean columns
        for col in column_names:
            if ref_data[col].dtype == "bool":
                ref_data[col] = ref_data[col].astype(float)
                analysis_data[col] = analysis_data[col].astype(float)

        return ref_data, analysis_data

    def _validate_data_set(self) -> None:
        """
        Validate that both reference and analysis data are set.

        Raises
        ------
        ValueError
            If reference or analysis data is not set
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Use set_reference_data() first.")

        if self.analysis_data is None:
            raise ValueError("Analysis data not set. Use set_analysis_data() first.")

    def _invalidate_results(self) -> None:
        """Mark all cached results as outdated."""
        self._univariate_outdated = True
        self._multivariate_outdated = True
        self._multivariate_outdated = True
