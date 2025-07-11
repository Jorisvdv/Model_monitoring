import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, brier_score_loss, confusion_matrix, roc_auc_score


class TemporalValidation:
    """
    A class for temporal validation and monitoring of machine learning model performance.

    This class provides comprehensive tools for monitoring model performance over time,
    including calculation of periodic performance metrics (AUC, Brier score), detection
    of performance alerts, and visualization of temporal trends in model behavior.

    The class supports multiple time granularities (yearly, monthly, weekly, daily) and
    can track multiple models simultaneously. It provides both tabular and visual outputs
    for performance monitoring and alerting.

    Parameters
    ----------
    models : dict
        Dictionary of trained models where keys are model names and values are
        fitted scikit-learn compatible models with predict_proba method.
    target_column : str
        Name of the target column in the data containing binary outcomes (0/1).
    date_column : str
        Name of the date column in the data for temporal grouping.
    rolling_window : int, default=1
        Rolling window size for temporal aggregation.
    chunk_period : {'Y', 'M', 'W', 'D'}, default='M'
        Time period for chunking data:
        - 'Y': Yearly
        - 'M': Monthly
        - 'W': Weekly
        - 'D': Daily
    auc_upper_limit : float, optional
        Upper threshold for AUC alerts. Performance above this triggers alerts.
    auc_lower_limit : float, optional
        Lower threshold for AUC alerts. Performance below this triggers alerts.
    brier_upper_limit : float, optional
        Upper threshold for Brier score alerts. Performance above this triggers alerts.
    brier_lower_limit : float, optional
        Lower threshold for Brier score alerts. Performance below this triggers alerts.

    Attributes
    ----------
    models : dict
        Dictionary of models being monitored.
    target_column : str
        Target column name.
    date_column : str
        Date column name.
    rolling_window : int
        Rolling window size.
    chunk_period : str
        Period for temporal chunking.
    period_name : str
        Human-readable name for the period (e.g., 'Month', 'Week').
    data : pd.DataFrame or None
        Stored data for monitoring.
    period_performance : dict
        Dictionary storing performance metrics by period.
    period_performance_df : pd.DataFrame
        DataFrame containing calculated periodic performance metrics.
    auc_upper_limit, auc_lower_limit : float or None
        AUC alert thresholds.
    brier_upper_limit, brier_lower_limit : float or None
        Brier score alert thresholds.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> import pandas as pd
    >>>
    >>> # Prepare models
    >>> models = {
    ...     'RF': RandomForestClassifier().fit(X_train, y_train),
    ...     'LR': LogisticRegression().fit(X_train, y_train)
    ... }
    >>>
    >>> # Initialize temporal validation
    >>> tv = TemporalValidation(
    ...     models=models,
    ...     target_column='outcome',
    ...     date_column='date',
    ...     chunk_period='M',
    ...     auc_lower_limit=0.7,
    ...     brier_upper_limit=0.3
    ... )
    >>>
    >>> # Add monitoring data
    >>> tv.append_data(monitoring_data)
    >>>
    >>> # Calculate performance metrics
    >>> performance_df = tv.calculate_period_performance()
    >>>
    >>> # Check for alerts
    >>> alerts = tv.get_performance_alerts()
    >>>
    >>> # Generate plots
    >>> fig = tv.plot_period_performance()
    >>> plt.show()
    """

    def __init__(
        self,
        models: Dict[str, Any],
        target_column: str,
        date_column: str,
        rolling_window: int = 1,
        chunk_period: Literal["Y", "M", "W", "D"] = "M",
        auc_upper_limit: Optional[float] = None,
        auc_lower_limit: Optional[float] = None,
        brier_upper_limit: Optional[float] = None,
        brier_lower_limit: Optional[float] = None,
    ) -> None:
        self.models = models
        self.target_column: str = target_column
        self.date_column: str = date_column
        self.rolling_window: int = rolling_window
        self.chunk_period: Literal["Y", "M", "W", "D"] = chunk_period
        period_rename = {
            "Y": "Year",
            "M": "Month",
            "W": "Week",
            "D": "Day",
        }
        self.period_name = period_rename[self.chunk_period]

        self.data: Optional[pd.DataFrame] = None
        self.initialize_performance_dict()
        self.auc_upper_limit = auc_upper_limit
        self.auc_lower_limit = auc_lower_limit
        self.brier_upper_limit = brier_upper_limit
        self.brier_lower_limit = brier_lower_limit

    def initialize_performance_dict(self) -> None:
        """
        Initialize the periodic performance dictionary dynamically for each model.

        Creates a dictionary structure to store performance metrics (AUC and Brier score)
        for each model across different time periods. The dictionary includes:
        - A list for storing period timestamps
        - Separate lists for AUC and Brier scores for each model

        This method is called during initialization and sets up the internal
        data structure used by calculate_period_performance().
        """

        self.period_performance = {self.period_name: []}
        for model_name in self.models.keys():
            self.period_performance[f"auc_{model_name}"] = []
            self.period_performance[f"brier_{model_name}"] = []

    def calculate_period_performance(
        self, data: Optional[pd.DataFrame], threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculate AUC and Brier score for each model on a periodic basis.

        This method resamples the input data according to the specified chunk_period
        and rolling_window, then calculates performance metrics for each model on
        each time period. Optionally calculates classification metrics when a
        threshold is provided.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Input data containing features, target, and date columns. If None,
            uses self.data if available.
        threshold : float, optional
            Classification threshold for binary predictions. When provided,
            calculates sensitivity, specificity, PPV, and NPV in addition to
            AUC and Brier score.

        Returns
        -------
        pd.DataFrame
            DataFrame with period index and columns for each model's performance:
            - auc_{model_name}: AUC values for each period
            - brier_{model_name}: Brier score values for each period
            - sensitivity_{model_name}: Sensitivity (if threshold provided)
            - specificity_{model_name}: Specificity (if threshold provided)
            - ppv_{model_name}: Positive predictive value (if threshold provided)
            - npv_{model_name}: Negative predictive value (if threshold provided)

        Raises
        ------
        ValueError
            If data is None and self.data is not available, or if required
            columns are missing from the data.

        Notes
        -----
        - Periods with no samples or only one class are skipped with warnings
        - The resulting DataFrame is stored in self.period_performance_df
        - Index is converted to PeriodIndex matching the chunk_period
        """
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for periodic performance calculation.")
            else:
                data = self.data
        if self.target_column not in data.columns or self.date_column not in data.columns:
            raise ValueError(
                f"Data must contain the target column '{self.target_column}' and date column '{self.date_column}'."
            )

        # Resample data on a periodic basis with rolling_window offset
        data = data.set_index(pd.to_datetime(data[self.date_column]))
        if self.chunk_period == "Y":
            resampled_data = data.resample(pd.offsets.YearEnd(self.rolling_window))
        elif self.chunk_period == "M":
            resampled_data = data.resample(pd.offsets.MonthEnd(self.rolling_window))
        elif self.chunk_period == "W":
            resampled_data = data.resample(pd.offsets.Week(self.rolling_window))
        elif self.chunk_period == "D":
            resampled_data = data.resample(pd.offsets.Day(self.rolling_window))
        else:
            raise ValueError(f"Unsupported chunk period: {self.chunk_period}")

        for period, group in resampled_data:
            X_period = group.drop(columns=[self.target_column, self.date_column])
            y_period = group[self.target_column]

            # Skip if no samples
            if len(X_period) == 0:
                logging.warning(
                    f"Skipping calculation for {self.period_name.lower()} {period} "
                    f"due to having no samples in this period."
                )
                continue

            # Skip if only one outcome (mostly in last month)
            if y_period.nunique() == 1:
                logging.warning(
                    f"Skipping calculation for {self.period_name.lower()} {period} due to having only one class."
                )
                continue

            self.period_performance[self.period_name].append(period)

            for model_name, model in self.models.items():
                y_pred = model.predict_proba(X_period)[:, 1]
                self.period_performance[f"auc_{model_name}"].append(roc_auc_score(y_period, y_pred))
                self.period_performance[f"brier_{model_name}"].append(brier_score_loss(y_period, y_pred))

                if threshold is not None:
                    y_pred_class = (y_pred >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_period, y_pred_class).ravel()

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

                    self.period_performance[f"sensitivity_{model_name}"].append(sensitivity)
                    self.period_performance[f"specificity_{model_name}"].append(specificity)
                    self.period_performance[f"ppv_{model_name}"].append(ppv)
                    self.period_performance[f"npv_{model_name}"].append(npv)

        self.period_performance_df = pd.DataFrame(self.period_performance)
        self.period_performance_df = self.period_performance_df.set_index(self.period_name)
        self.period_performance_df.index = self.period_performance_df.index.to_period(self.chunk_period)

        return self.period_performance_df

    def append_data(self, data: pd.DataFrame) -> None:
        """
        Append new data to the existing data store for temporal monitoring.

        This method validates and appends new monitoring data to the internal
        data store. It performs comprehensive validation to ensure data consistency
        including column matching and data type compatibility.

        Parameters
        ----------
        data : pd.DataFrame
            New data to append containing features, target column, and date column.
            Must have the same column structure and types as existing data.

        Raises
        ------
        ValueError
            If the date_column or target_column are missing from the new data,
            or if column names or data types don't match existing data.

        Notes
        -----
        - If no existing data, initializes self.data with the new data
        - Automatically converts date_column to datetime format
        - Validates column names and data types for consistency
        - Uses pd.concat to append data while preserving data types

        Examples
        --------
        >>> tv = TemporalValidation(models, 'target', 'date')
        >>> tv.append_data(new_monitoring_data)
        >>> print(f"Total samples: {len(tv.data)}")
        """
        # Check if new data matches columns and types of existing data
        if self.data is None:
            logging.warning("No existing data found. Initializing with new data.")
            # Check if date_column is in data
            if self.date_column not in data.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in the new data.")
            # Convert date_column to datetime if it exists
            data[self.date_column] = pd.to_datetime(data[self.date_column], errors="coerce")
            # Check if target_column is in data
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the new data.")
            # Initialize self.data with the new data
            self.data = data.copy()
        else:
            if not self.data.columns.equals(data.columns):
                # Check mismatches in columns
                column_mismatches = self.data.columns[~self.data.columns.isin(data.columns)]
                raise ValueError(f"New data does not match existing data columns: {column_mismatches.tolist()}")
            if not self.data.dtypes.equals(data.dtypes):
                # Check mismatches in data types
                # Change date columns to datetime if they are not already
                if self.date_column in data.columns:
                    data[self.date_column] = pd.to_datetime(data[self.date_column], errors="coerce")
                dtype_mismatches = self.data.dtypes[~self.data.dtypes.isin(data.dtypes)]
                if not dtype_mismatches.empty:
                    raise ValueError(f"New data does not match existing data types: {dtype_mismatches.tolist()}")

            self.data = pd.concat([self.data, data], ignore_index=True)

    def get_performance_alerts(
        self,
        auc_upper_limit: Optional[float] = None,
        auc_lower_limit: Optional[float] = None,
        brier_upper_limit: Optional[float] = None,
        brier_lower_limit: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Check for performance alerts based on AUC and Brier score limits.

        Identifies periods where model performance falls outside specified thresholds
        for either AUC or Brier score metrics. Returns a structured DataFrame with
        alert details for further analysis or reporting.

        Parameters
        ----------
        auc_upper_limit : float, optional
            Upper threshold for AUC alerts. Values above this trigger alerts.
            If None, uses instance-level auc_upper_limit.
        auc_lower_limit : float, optional
            Lower threshold for AUC alerts. Values below this trigger alerts.
            If None, uses instance-level auc_lower_limit.
        brier_upper_limit : float, optional
            Upper threshold for Brier score alerts. Values above this trigger alerts.
            If None, uses instance-level brier_upper_limit.
        brier_lower_limit : float, optional
            Lower threshold for Brier score alerts. Values below this trigger alerts.
            If None, uses instance-level brier_lower_limit.

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with alerts information:
            - Index: ('type', 'period') where type is 'ROAUC' or 'Brier'
            - Columns: 'value', 'model', 'threshold', 'direction'
            Empty DataFrame if no alerts found.

        Raises
        ------
        ValueError
            If no data is available or if all threshold limits are None.

        Notes
        -----
        - Automatically calculates period performance if not already done
        - Uses instance-level thresholds as defaults when parameters are None
        - Direction indicates whether alert is for 'lower' or 'higher' than threshold
        - Supports multiple models and metric types simultaneously

        Examples
        --------
        >>> alerts = tv.get_performance_alerts(
        ...     auc_lower_limit=0.7,
        ...     brier_upper_limit=0.3
        ... )
        >>> print(f"Found {len(alerts)} alerts")
        >>> print(alerts[['model', 'value', 'threshold']])
        """
        if not hasattr(self, "period_performance_df"):
            if self.data is None:
                raise ValueError("No data available. Please append data first.")
            else:
                self.calculate_period_performance(self.data)

        if auc_upper_limit is None:
            auc_upper_limit = self.auc_upper_limit
        if auc_lower_limit is None:
            auc_lower_limit = self.auc_lower_limit
        if brier_upper_limit is None:
            brier_upper_limit = self.brier_upper_limit
        if brier_lower_limit is None:
            brier_lower_limit = self.brier_lower_limit

        if all(limit is None for limit in [auc_upper_limit, auc_lower_limit, brier_upper_limit, brier_lower_limit]):
            raise ValueError("One of the limits should be defined to output performance alerts")

        alert_dfs = []
        if auc_lower_limit is not None or auc_upper_limit is not None:
            for model_name in self.models.keys():
                auc_col = f"auc_{model_name}"
                if auc_col in self.period_performance_df.columns:
                    if auc_lower_limit is not None:
                        mask = self.period_performance_df[auc_col] < auc_lower_limit
                        auc_lower_values = self.period_performance_df[auc_col][mask]
                        auc_lower_values.name = "value"
                        auc_lower = pd.DataFrame(auc_lower_values)
                        auc_lower["type"] = "ROAUC"
                        auc_lower["model"] = model_name
                        auc_lower["threshold"] = auc_lower_limit
                        auc_lower["direction"] = "lower"
                        alert_dfs.append(auc_lower)
                    if auc_upper_limit is not None:
                        mask = self.period_performance_df[auc_col] > auc_upper_limit
                        auc_upper_values = self.period_performance_df[auc_col][mask]
                        auc_upper_values.name = "value"
                        auc_upper = pd.DataFrame(auc_upper_values)
                        auc_upper["type"] = "ROAUC"
                        auc_upper["model"] = model_name
                        auc_upper["threshold"] = auc_upper_limit
                        auc_upper["direction"] = "higher"
                        alert_dfs.append(auc_upper)

        if brier_lower_limit is not None or brier_upper_limit is not None:
            for model_name in self.models.keys():
                brier_col = f"brier_{model_name}"
                if brier_col in self.period_performance_df.columns:
                    if brier_lower_limit is not None:
                        mask = self.period_performance_df[brier_col] < brier_lower_limit
                        brier_lower_values = self.period_performance_df[brier_col][mask]
                        brier_lower_values.name = "value"
                        brier_lower = pd.DataFrame(brier_lower_values)
                        brier_lower["type"] = "Brier"
                        brier_lower["model"] = model_name
                        brier_lower["threshold"] = brier_lower_limit
                        brier_lower["direction"] = "lower"
                        alert_dfs.append(brier_lower)
                    if brier_upper_limit is not None:
                        mask = self.period_performance_df[brier_col] > brier_upper_limit
                        brier_upper_values = self.period_performance_df[brier_col][mask]
                        brier_upper_values.name = "value"
                        brier_upper = pd.DataFrame(brier_upper_values)
                        brier_upper["type"] = "Brier"
                        brier_upper["model"] = model_name
                        brier_upper["threshold"] = brier_upper_limit
                        brier_upper["direction"] = "higher"
                        alert_dfs.append(brier_upper)

        if alert_dfs:
            merged_df = pd.concat(alert_dfs)
            merged_df = merged_df.set_index("type", append=True)
            merged_df = merged_df.swaplevel(0, 1)
            return merged_df
        else:
            # Return empty DataFrame with correct structure when no alerts found
            alerts_df = pd.DataFrame(columns=["type", "model", "period", "value", "threshold", "direction"])
            alerts_df = alerts_df.set_index(["type", "period"])
            return alerts_df

    def plot_period_performance(
        self,
        title: Optional[str] = None,
        auc_upper_limit: Optional[float] = None,
        auc_lower_limit: Optional[float] = None,
        brier_upper_limit: Optional[float] = None,
        brier_lower_limit: Optional[float] = None,
        plot_roauc: bool = True,
        plot_brier: bool = True,
        remove_last_period: bool = False,
        mark_alerts: bool = True,
        show_limits: bool = True,
        predefined_axes: Optional[List[Axes]] = None,
    ) -> Union[List[Axes], Figure]:
        """
        Plot periodic performance metrics for all models over time.

        Creates time series plots showing AUC and/or Brier score evolution
        across the specified time periods. Optionally highlights performance
        alerts and threshold limits for monitoring purposes.

        Parameters
        ----------
        title : str, optional
            Custom title for the plot. If None, generates automatic title
            based on chunk_period.
        auc_upper_limit, auc_lower_limit : float, optional
            AUC alert thresholds for visualization. If None, uses instance defaults.
        brier_upper_limit, brier_lower_limit : float, optional
            Brier score alert thresholds for visualization. If None, uses instance defaults.
        plot_roauc : bool, default=True
            Whether to plot AUC performance subplot.
        plot_brier : bool, default=True
            Whether to plot Brier score performance subplot.
        remove_last_period : bool, default=False
            Whether to exclude the most recent period (useful for incomplete data).
        mark_alerts : bool, default=True
            Whether to highlight alert points with red diamonds.
        show_limits : bool, default=True
            Whether to show threshold limit lines.
        predefined_axes : list of matplotlib.axes.Axes, optional
            Pre-existing axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.figure.Figure or list of matplotlib.axes.Axes
            If predefined_axes is None, returns Figure object.
            If predefined_axes provided, returns the axes list.

        Raises
        ------
        ValueError
            If no data is available for plotting.

        Notes
        -----
        - Automatically calculates performance if not already done
        - Each model gets a different line color/style
        - Alert points are marked with red diamond markers
        - Threshold lines are shown as red dashed horizontal lines
        - Grid is enabled for better readability

        Examples
        --------
        >>> fig = tv.plot_period_performance(
        ...     title="Model Performance Over Time",
        ...     plot_brier=False,
        ...     mark_alerts=True
        ... )
        >>> plt.savefig("performance_plot.png")
        """
        if title is None:
            if self.chunk_period == "D":
                title = "Daily Performance"
            else:
                title = f"{self.period_name.capitalize()}ly Performance"
        if not hasattr(self, "period_performance_df") or self.period_performance_df is None:
            if self.data is None:
                raise ValueError("No data available. Please append data first.")
            else:
                logging.info("Calculating periodic performance from data.")
                self.calculate_period_performance(self.data)

        df = self.period_performance_df.copy()
        if remove_last_period:
            df = df.iloc[:-1]

        nrows = int(plot_roauc) + int(plot_brier)
        if predefined_axes is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 4 * nrows), sharex=True)
            # Ensure axes is always a list for consistent indexing
            if nrows == 1:
                axes = [axes]
        else:
            axes = predefined_axes

        ax_idx = 0
        periods = df.index

        # Get months with alerts
        if auc_upper_limit is None:
            auc_upper_limit = self.auc_upper_limit
        if auc_lower_limit is None:
            auc_lower_limit = self.auc_lower_limit
        if brier_upper_limit is None:
            brier_upper_limit = self.brier_upper_limit
        if brier_lower_limit is None:
            brier_lower_limit = self.brier_lower_limit
        if mark_alerts:
            alerts = self.get_performance_alerts(
                auc_upper_limit=auc_upper_limit,
                auc_lower_limit=auc_lower_limit,
                brier_upper_limit=brier_upper_limit,
                brier_lower_limit=brier_lower_limit,
            )

        # Plot AUC
        if plot_roauc:
            ax = axes[ax_idx]
            for model_name in self.models.keys():
                auc_col = f"auc_{model_name}"
                if auc_col in df.columns:
                    ax.plot(periods.to_timestamp(), df[auc_col], marker="o", label=model_name)
                    # Mark alerts
                    if mark_alerts and (auc_lower_limit is not None or auc_upper_limit is not None):
                        if "ROAUC" in alerts.index.levels[0]:
                            alert_periods = alerts.xs("ROAUC", level="type").index.to_timestamp()
                            alert_values = alerts.xs("ROAUC", level="type")["value"].values
                            ax.scatter(
                                alert_periods,
                                alert_values,
                                color="red",
                                marker="D",
                                s=80,
                                label=f"{model_name} Alert" if ax_idx == 0 else None,
                            )
            if show_limits:
                if auc_upper_limit is not None:
                    ax.axhline(y=auc_upper_limit, color="r", linestyle="--", label="AUC Upper Limit")
                if auc_lower_limit is not None:
                    ax.axhline(y=auc_lower_limit, color="r", linestyle="--", label="AUC Lower Limit")
            ax.set_ylabel("ROAUC")
            if len(axes) == 1:
                ax.set_xlabel("Date")
            ax.set_title(title)
            ax.legend(loc="upper right")
            ax.grid(True)
            ax_idx += 1

        # Plot Brier
        if plot_brier:
            ax = axes[ax_idx]
            for model_name in self.models.keys():
                brier_col = f"brier_{model_name}"
                if brier_col in df.columns:
                    ax.plot(periods.to_timestamp(), df[brier_col], marker="o", label=model_name)
                    # Mark alerts
                    if mark_alerts and (brier_lower_limit is not None or brier_upper_limit is not None):
                        if "Brier" in alerts.index.levels[0]:
                            alert_periods = alerts.xs("Brier", level="type").index.to_timestamp()
                            alert_values = alerts.xs("Brier", level="type")["value"].values
                            ax.scatter(
                                alert_periods,
                                alert_values,
                                color="red",
                                marker="D",
                                s=80,
                                label=f"{model_name} Alert",
                            )
            if show_limits:
                if brier_upper_limit is not None:
                    ax.axhline(y=brier_upper_limit, color="r", linestyle="--", label="Brier Upper Limit")
                if brier_lower_limit is not None:
                    ax.axhline(y=brier_lower_limit, color="r", linestyle="--", label="Brier Lower Limit")
            ax.set_ylabel("Brier Score")
            ax.set_xlabel("Date")
            ax.legend(loc="upper right")
            ax.grid(True)

        if predefined_axes is not None:
            return axes
        else:
            fig.tight_layout()
            return fig

    def plot_roc_curves(self, data: Optional[pd.DataFrame] = None, spec: str = "") -> Tuple[Figure, Axes]:
        """
        Plot ROC curves for all models on the same axes.

        Creates a single plot with ROC curves for each model in the monitoring
        set, useful for comparing discriminative performance across models.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to use for plotting. If None, uses self.data.
        spec : str, default=""
            Additional specification text to append to the plot title.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes objects for further customization.

        Raises
        ------
        ValueError
            If no data is available for plotting.

        Notes
        -----
        - Each model gets a different color and is labeled by name
        - Diagonal reference line (random classifier) is automatically included
        - Plot limits are set to [0, 1] for both axes
        - Uses sklearn.metrics.RocCurveDisplay for consistent formatting

        Examples
        --------
        >>> fig, ax = tv.plot_roc_curves(test_data, spec="(Test Set)")
        >>> ax.set_title("Model Comparison - ROC Curves")
        >>> plt.show()
        """
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data

        fig, ax = plt.subplots()
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        for model_name, model in self.models.items():
            RocCurveDisplay.from_estimator(model, X, y, name=model_name, ax=ax)
        ax.set_title(f"ROC Curves {spec}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    def plot_calibration_curves(self, data: Optional[pd.DataFrame] = None, spec: str = "") -> Tuple[Figure, Axes]:
        """
        Plot calibration curves for all models on the same axes.

        Creates reliability diagrams showing how well-calibrated the predicted
        probabilities are for each model. Well-calibrated models should have
        points close to the diagonal line.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to use for plotting. If None, uses self.data.
        spec : str, default=""
            Additional specification text to append to the plot title.

        Returns
        -------
        tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
            The figure and axes objects for further customization.

        Raises
        ------
        ValueError
            If no data is available for plotting.

        Notes
        -----
        - Each model gets a different color and is labeled by name
        - Perfect calibration line (diagonal) is automatically included
        - Uses 20 bins for calibration calculation
        - Plot limits are set to [0, 1] for both axes
        - Uses sklearn.calibration.CalibrationDisplay for consistent formatting

        Examples
        --------
        >>> fig, ax = tv.plot_calibration_curves(validation_data, spec="(Validation)")
        >>> ax.set_title("Model Calibration Analysis")
        >>> plt.show()
        """
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data

        fig, ax = plt.subplots()
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        for model_name, model in self.models.items():
            CalibrationDisplay.from_estimator(model, X, y, n_bins=20, name=model_name, ax=ax)
        ax.set_title(f"Calibration Curves {spec}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig

    def plot_combined_curves(
        self,
        data: Optional[pd.DataFrame] = None,
        spec: str = "",
        plot_roauc: bool = True,
        plot_brier: bool = False,
        auc_upper_limit: Optional[float] = None,
        auc_lower_limit: Optional[float] = None,
        brier_upper_limit: Optional[float] = None,
        brier_lower_limit: Optional[float] = None,
        remove_last_period: bool = False,
        mark_alerts: bool = False,
        show_limits: bool = True,
    ) -> Figure:
        """
        Create a comprehensive combined visualization with calibration and temporal performance.

        Generates a two-panel figure combining:
        - Left panel: Calibration plot with histogram of predicted probabilities
        - Right panel: Temporal performance plot showing metrics over time

        This provides a complete view of both model calibration quality and
        temporal stability in a single visualization.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to use for plotting. If None, uses self.data.
        spec : str, default=""
            Additional specification text to append to plot titles.
        plot_roauc : bool, default=True
            Whether to include AUC in the temporal performance plot.
        plot_brier : bool, default=False
            Whether to include Brier score in the temporal performance plot.
        auc_upper_limit, auc_lower_limit : float, optional
            AUC alert thresholds for temporal plot.
        brier_upper_limit, brier_lower_limit : float, optional
            Brier score alert thresholds for temporal plot.
        remove_last_period : bool, default=False
            Whether to exclude the most recent period from temporal plot.
        mark_alerts : bool, default=False
            Whether to highlight alert points in temporal plot.
        show_limits : bool, default=True
            Whether to show threshold lines in temporal plot.

        Returns
        -------
        matplotlib.figure.Figure
            The complete figure object with both panels.

        Raises
        ------
        ValueError
            If no data is available for plotting.

        Notes
        -----
        - Left panel (A): Shows calibration curves and prediction histograms
        - Right panel (B): Shows temporal performance trends
        - Uses a sophisticated grid layout for optimal space utilization
        - Calibration plot includes histogram of predicted probabilities
        - Temporal plot can show either or both AUC and Brier score metrics
        - Figure size is optimized for publication quality (14x8 inches)

        Examples
        --------
        >>> fig = tv.plot_combined_curves(
        ...     data=monitoring_data,
        ...     spec="(Production Data)",
        ...     plot_roauc=True,
        ...     plot_brier=True,
        ...     mark_alerts=True
        ... )
        >>> fig.savefig("comprehensive_model_analysis.png", dpi=300, bbox_inches='tight')
        """
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(nrows=10, ncols=12)

        ax_calibration_curve = fig.add_subplot(gs[:7, :4])
        ax_hist = fig.add_subplot(gs[8:10, :4], sharex=ax_calibration_curve)

        # 1. Calibration and Histogram plot
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        for model_name, model in self.models.items():
            display: CalibrationDisplay = CalibrationDisplay.from_estimator(
                model,
                X,
                y,
                n_bins=20,
                name=model_name,
                ax=ax_calibration_curve,
            )
            ax_hist.hist(
                display.y_prob,
                range=(0, 1),
                bins=20,
                label=model_name,
                alpha=0.7,
            )

        ax_calibration_curve.grid(True)
        ax_calibration_curve.set_title(f"A: Calibration plot {spec}")
        ax_calibration_curve.set_xlabel("Predicted Probability")
        ax_calibration_curve.set_ylabel("Fraction of Positives")
        ax_calibration_curve.legend(loc="upper left")
        ax_calibration_curve.spines["top"].set_visible(False)
        ax_calibration_curve.spines["right"].set_visible(False)
        ax_calibration_curve.set_ylim(0, 1)
        ax_calibration_curve.set_xlim(0, 1)

        ax_hist.set(title="Histogram", xlabel="Predicted probability", ylabel="Count")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.legend()

        # 2. Periodic performance plot
        if self.period_performance_df is None:
            if self.data is None:
                # This should not trigger due to check above, but for redundancy
                raise ValueError("No data available. Please append data first.")
            else:
                logging.info("Calculating periodic performance from data.")
                # Calculate performance if not already done
                self.calculate_period_performance(data)
        periodic_axes = []
        if plot_roauc:
            if not plot_brier:
                ax_roauc = fig.add_subplot(gs[:, 4:])
                periodic_axes.append(ax_roauc)
            else:
                ax_roauc = fig.add_subplot(gs[:5, 4:])
                periodic_axes.append(ax_roauc)
                ax_bier = fig.add_subplot(gs[5:, 4:], sharex=ax_roauc)
                periodic_axes.append(ax_bier)
        elif plot_brier:
            ax_bier = fig.add_subplot(gs[:, 4:])
            periodic_axes.append(ax_bier)

        if self.chunk_period == "D":
            period_title = f"Daily model performance {spec}"
        else:
            period_title = f"B: {self.period_name.capitalize()}ly model performance {spec}"
        self.plot_period_performance(
            plot_roauc=plot_roauc,
            plot_brier=plot_brier,
            predefined_axes=periodic_axes,
            title=period_title,
            auc_upper_limit=auc_upper_limit,
            auc_lower_limit=auc_lower_limit,
            brier_upper_limit=brier_upper_limit,
            brier_lower_limit=brier_lower_limit,
            remove_last_period=remove_last_period,
            mark_alerts=mark_alerts,
            show_limits=show_limits,
        )

        fig.tight_layout()
        return fig
