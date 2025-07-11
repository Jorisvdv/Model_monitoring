import logging
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, brier_score_loss, roc_auc_score, confusion_matrix


class TemporalValidation:
    def __init__(
        self,
        models: dict,
        target_column: str,
        date_column: str,
        rolling_window: int = 1,
        chunk_period: Literal["Y", "M", "W", "D"] = "M",
        auc_upper_limit: float | None = None,
        auc_lower_limit: float | None = None,
        brier_upper_limit: float | None = None,
        brier_lower_limit: float | None = None,
    ):
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

        self.data: pd.DataFrame | None = None
        self.initialize_performance_dict()
        self.auc_upper_limit = auc_upper_limit
        self.auc_lower_limit = auc_lower_limit
        self.brier_upper_limit = brier_upper_limit
        self.brier_lower_limit = brier_lower_limit

        # self.models_to_plot = []
    def initialize_performance_dict(self):
        """Initialize the periodic performance dictionary dynamically for each model."""
        

        self.period_performance = {self.period_name: []}
        for model_name in self.models.keys():
            self.period_performance[f"auc_{model_name}"] = []
            self.period_performance[f"brier_{model_name}"] = []

    def calculate_period_performance(self, data: Optional[pd.DataFrame], threshold: float | None = None) -> pd.DataFrame:
        """
        Calculate AUC and Brier score for each model on a periodic basis.
        Outputs sensitivity, specificity, PPV, and NPV if threshold is provided.

        """
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for periodic performance calculation.")
            else:
                data = self.data
        if not self.target_column in data.columns or not self.date_column in data.columns:
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

            # Skip if only one outcome (mostly in last month)
            if y_period.nunique() == 1:
                logging.warning(f"Skipping AUC calculation for {self.period_name.lower()} {period} due to having only one class.")
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
        # data[self.date_column] = pd.to_datetime(data[self.date_column])
        # data = data.sort_values(by=self.date_column)
        # data["month"] = data[self.date_column].dt.to_period(self.chunk_period)

        # monthly_performance = []
        # for month in data["month"].unique():
        #     end_date = month.to_timestamp() + MonthEnd(0)
        #     start_date = end_date - pd.DateOffset(months=self.rolling_window - 1)

        #     monthly_data = data[(data[self.date_column] >= start_date) & (data[self.date_column] <= end_date)]

        #     if len(monthly_data) > 0:
        #         X_month = monthly_data.drop(columns=[self.target_column, "month", self.date_column])
        #         y_month = monthly_data[self.target_column]

        #         for name, model in self.models.items():
        #             y_pred_proba = model.predict_proba(X_month)[:, 1]
        #             if y_month.nunique() < 2:
        #                 auc = np.nan
        #                 logging.warning(f"Skipping AUC calculation for month {month} due to having only one class.")
        #                 continue

        #             auc = roc_auc_score(y_month, y_pred_proba)

        #             brier = brier_score_loss(y_month, y_pred_proba)
        #             monthly_performance.append(
        #                 {"month": month.to_timestamp(), "model": name, "AUC": auc, "Brier": brier}
        #             )

        return self.period_performance_df
    
    def append_data(self, data: pd.DataFrame):
        """Append new data to the existing data."""
        # Check if new data matches columns and types of existing data
        if self.data is None:
            logging.warning("No existing data found. Initializing with new data.")
            # Check if date_column is in data
            if self.date_column not in data.columns:
                raise ValueError(f"Date column '{self.date_column}' not found in the new data.")
            # Convert date_column to datetime if it exists
            data[self.date_column] = pd.to_datetime(data[self.date_column], errors='coerce')
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
                    data[self.date_column] = pd.to_datetime(data[self.date_column], errors='coerce')
                dtype_mismatches = self.data.dtypes[~self.data.dtypes.isin(data.dtypes)]
                if not dtype_mismatches.empty:
                    raise ValueError(f"New data does not match existing data types: {dtype_mismatches.tolist()}")

            self.data = pd.concat([self.data, data], ignore_index=True)

    def get_performance_alerts(self):
        """Check for performance alerts based on AUC and Brier score limits."""
        if self.period_performance_df is None:
            if self.data is None:
                raise ValueError("No data available. Please append data first.")
            else:
                self.calculate_period_performance(self.data)

        alerts = []
        if self.auc_lower_limit is not None or self.auc_upper_limit is not None:
            for model_name in self.models.keys():
                auc_col = f"auc_{model_name}"
                if auc_col in self.period_performance_df.columns:
                    if self.auc_lower_limit is not None:
                        mask = self.period_performance_df[auc_col] < self.auc_lower_limit
                        for idx in self.period_performance_df[mask].index:
                            alerts.append({
                                "type": "AUC",
                                "model": model_name,
                                "period": self.period_performance_df[self.period_name][idx],
                                "value": self.period_performance_df[auc_col][idx],
                                "threshold": self.auc_lower_limit,
                                "direction": "lower"
                            })
                    if self.auc_upper_limit is not None:
                        mask = self.period_performance_df[auc_col] > self.auc_upper_limit
                        for idx in self.period_performance_df[mask].index:
                            alerts.append({
                                "type": "AUC",
                                "model": model_name,
                                "period": self.period_performance_df[self.period_name][idx],
                                "value": self.period_performance_df[auc_col][idx],
                                "threshold": self.auc_upper_limit,
                                "direction": "higher"
                            })

        if self.brier_lower_limit is not None or self.brier_upper_limit is not None:
            for model_name in self.models.keys():
                brier_col = f"brier_{model_name}"
                if brier_col in self.period_performance_df.columns:
                    if self.brier_lower_limit is not None:
                        mask = self.period_performance_df[brier_col] < self.brier_lower_limit
                        for idx in self.period_performance_df[mask].index:
                            alerts.append({
                                "type": "Brier",
                                "model": model_name,
                                "period": self.period_performance_df[self.period_name][idx],
                                "value": self.period_performance_df[brier_col][idx],
                                "threshold": self.brier_lower_limit,
                                "direction": "lower"
                            })
                    if self.brier_upper_limit is not None:
                        mask = self.period_performance_df[brier_col] > self.brier_upper_limit
                        for idx in self.period_performance_df[mask].index:
                            alerts.append({
                                "type": "Brier",
                                "model": model_name,
                                "period": self.period_performance_df[self.period_name][idx],
                                "value": self.period_performance_df[brier_col][idx],
                                "threshold": self.brier_upper_limit,
                                "direction": "higher"
                            })

        return alerts

    def plot_period_performance(
        self,
        title: str = f"{self.period_name.capitalize()}ly Performance",
        auc_upper_limit: float = None,
        auc_lower_limit: float = None,
        brier_upper_limit: float = None,
        brier_lower_limit: float = None,
        plot_auc: bool = True,
        plot_brier: bool = True,
        remove_last_period: bool = False,
        mark_alerts: bool = True,
        show_limits: bool = True,  # New option to show/hide limits
    ):
        if not hasattr(self, "period_performance_df") or self.period_performance_df is None:
            if self.data is None:
                raise RuntimeError("No data available. Please append data first.")
            else:
                logging.info("Calculating periodic performance from data.")
                self.calculate_period_performance(self.data)

        df = self.period_performance_df.copy()
        if remove_last_period:
            df = df.iloc[:-1]

        nrows = int(plot_auc) + int(plot_brier)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 4 * nrows), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        ax_idx = 0
        periods = df[self.period_name]

        # Plot AUC
        if plot_auc:
            ax = axes[ax_idx]
            for model_name in self.models.keys():
                auc_col = f"auc_{model_name}"
                if auc_col in df.columns:
                    ax.plot(periods, df[auc_col], marker="o", label=model_name)
                    # Mark alerts
                    if mark_alerts and (auc_lower_limit is not None or auc_upper_limit is not None):
                        alert_mask = pd.Series([False] * len(df))
                        # Check both instance (method argument) and class (self) limits
                        if auc_lower_limit is not None:
                            alert_mask |= df[auc_col] < auc_lower_limit
                        elif self.auc_lower_limit is not None:
                            alert_mask |= df[auc_col] < self.auc_lower_limit
                        if auc_upper_limit is not None:
                            alert_mask |= df[auc_col] > auc_upper_limit
                        elif self.auc_upper_limit is not None:
                            alert_mask |= df[auc_col] > self.auc_upper_limit
                        alert_periods = periods[alert_mask]
                        alert_values = df[auc_col][alert_mask]
                        ax.scatter(alert_periods, alert_values, color="red", marker="D", s=80, label=f"{model_name} Alert" if ax_idx == 0 else None)
            if show_limits:
                if auc_upper_limit is not None:
                    ax.axhline(y=auc_upper_limit, color="r", linestyle="--", label="AUC Upper Limit")
                if auc_lower_limit is not None:
                    ax.axhline(y=auc_lower_limit, color="r", linestyle="--", label="AUC Lower Limit")
            ax.set_ylabel("AUC")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            ax_idx += 1

        # Plot Brier
        if plot_brier:
            ax = axes[ax_idx]
            for model_name in self.models.keys():
                brier_col = f"brier_{model_name}"
                if brier_col in df.columns:
                    ax.plot(periods, df[brier_col], marker="o", label=model_name)
                    # Mark alerts
                    if mark_alerts and (brier_lower_limit is not None or brier_upper_limit is not None):
                        alert_mask = pd.Series([False] * len(df))
                        if brier_lower_limit is not None:
                            alert_mask |= df[brier_col] < brier_lower_limit
                        elif self.brier_lower_limit is not None:
                            alert_mask |= df[brier_col] < self.brier_lower_limit
                        if brier_upper_limit is not None:
                            alert_mask |= df[brier_col] > brier_upper_limit
                        elif self.brier_upper_limit is not None:
                            alert_mask |= df[brier_col] > self.brier_upper_limit
                        alert_periods = periods[alert_mask]
                        alert_values = df[brier_col][alert_mask]
                        ax.scatter(alert_periods, alert_values, color="red", marker="D", s=80, label=f"{model_name} Alert" if ax_idx == 0 else None)
            if show_limits:
                if brier_upper_limit is not None:
                    ax.axhline(y=brier_upper_limit, color="r", linestyle="--", label="Brier Upper Limit")
                if brier_lower_limit is not None:
                    ax.axhline(y=brier_lower_limit, color="r", linestyle="--", label="Brier Lower Limit")
            ax.set_ylabel("Brier Score")
            ax.set_xlabel(self.period_name.capitalize())
            ax.legend()
            ax.grid(True)

        fig.tight_layout()
        return fig


    def plot_roc_curves(self, data: Optional[pd.DataFrame] = None, spec=""):
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data

        fig, ax = plt.subplots()
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        for model_name, model in self.models.items():
            RocCurveDisplay.from_estimator(
                model, X, y, name=model_name, ax=ax
            )
        ax.set_title(f"ROC Curves {spec}")
        return fig, ax

    def plot_calibration_curves(self, data: Optional[pd.DataFrame] = None, spec=""):
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data
        
        fig, ax = plt.subplots()
        X = data.drop(columns=[self.target_column, self.date_column])
        y = data[self.target_column]
        for model_name, model in self.models.items():
            CalibrationDisplay.from_estimator(
                model, X, y, n_bins=20, name=model_name, ax=ax
            )
        ax.set_title(f"Calibration Curves {spec}")
        return fig, ax

    def plot_combined_curves(self, data: Optional[pd.DataFrame] = None, spec=""):
        if data is None:
            if self.data is None:
                raise ValueError("Data must be provided for plotting when no internal data is available.")
            data = self.data
        
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(nrows=11, ncols=12)

        ax_calibration_curve = fig.add_subplot(gs[:8, :4])
        ax_hist = fig.add_subplot(gs[8:10, :4], sharex=ax_calibration_curve)
        ax_perf = fig.add_subplot(gs[:, 5:])

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

        ax_hist.set(title="Histogram", xlabel="Predicted probability", ylabel="Count")
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        ax_hist.legend()

        # 2. Periodic performance plot
        if self.period_performance_df is None:
            if self.data is None:
                raise RuntimeError("No data available. Please append data first.")
            else:
                logging.info("Calculating periodic performance from data.")
                # Calculate performance if not already done
                self.calculate_period_performance(data)
        performance_df = self.period_performance_df
        plot_df = performance_df.set_index(self.period_name)

        for model_name in self.models.keys():
            plot_df[f"auc_{model_name}"].plot(ax=ax_perf, label=model_name)

        if self.auc_lower_limit:
            ax_perf.axhline(
                self.auc_lower_limit,
                color="red",
                linestyle="--",
                alpha=0.5,
                label="AUROC Lower Limit",
            )

        ax_perf.set_title(f"B: {self.period_name.capitalize()}ly model performance {spec}")
        ax_perf.set_xlabel(self.period_name.capitalize())
        ax_perf.set_ylabel("AUROC")
        ax_perf.legend()
        ax_perf.spines["top"].set_visible(False)
        ax_perf.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig
