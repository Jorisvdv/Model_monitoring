# %% [markdown]
# # Model results

import itertools
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import RocCurveDisplay, brier_score_loss, roc_auc_score


class TemporalValidation:
    def __init__(self, models: dict, target_column: str, date_column: str, rolling_window: int = 1):
        self.models = models
        self.target_column = target_column
        self.date_column = date_column
        self.rolling_window = rolling_window
        self.monthly_performance_df = None
        self.models_to_plot = []

    def calculate_monthly_performance(self, data: pd.DataFrame):
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.sort_values(by=self.date_column)
        data["month"] = data[self.date_column].dt.to_period("M")

        monthly_performance = []
        for month in data["month"].unique():
            end_date = month.to_timestamp() + MonthEnd(0)
            start_date = end_date - pd.DateOffset(months=self.rolling_window - 1)

            monthly_data = data[(data[self.date_column] >= start_date) & (data[self.date_column] <= end_date)]

            if len(monthly_data) > 0:
                X_month = monthly_data.drop(columns=[self.target_column, "month", self.date_column])
                y_month = monthly_data[self.target_column]

                for name, model in self.models.items():
                    y_pred_proba = model.predict_proba(X_month)[:, 1]
                    auc = roc_auc_score(y_month, y_pred_proba)
                    brier = brier_score_loss(y_month, y_pred_proba)
                    monthly_performance.append(
                        {"month": month.to_timestamp(), "model": name, "AUC": auc, "Brier": brier}
                    )

        self.monthly_performance_df = pd.DataFrame(monthly_performance)
        return self.monthly_performance_df

    def plot_monthly_performance(
        self,
        title,
        auc_upper_limit=None,
        auc_lower_limit=None,
        brier_upper_limit=None,
        brier_lower_limit=None,
        plot_auc=True,
        plot_brier=True,
        remove_last_month=False,
    ):
        if self.monthly_performance_df is None:
            print("Please run calculate_monthly_performance() first.")
            return None

        df = self.monthly_performance_df.copy()
        if remove_last_month:
            df = df.iloc[:-1]

        fig, axes = plt.subplots(nrows=2 if plot_auc and plot_brier else 1, ncols=1, figsize=(12, 8), sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        ax_idx = 0
        if plot_auc:
            ax = axes[ax_idx]
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                ax.plot(model_df["month"], model_df["AUC"], marker="o", label=model)
            if auc_upper_limit:
                ax.axhline(y=auc_upper_limit, color="r", linestyle="--", label="Upper Limit")
            if auc_lower_limit:
                ax.axhline(y=auc_lower_limit, color="r", linestyle="--", label="Lower Limit")
            ax.set_ylabel("AUC")
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
            # Check for alerts
            if auc_lower_limit or auc_upper_limit:
                alerts = df[(df["AUC"] < auc_lower_limit) | (df["AUC"] > auc_upper_limit)]
                if not alerts.empty:
                    print("AUC Alerts:")
                    print(alerts)

            ax_idx += 1

        if plot_brier:
            ax = axes[ax_idx]
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                ax.plot(model_df["month"], model_df["Brier"], marker="o", label=model)
            if brier_upper_limit:
                ax.axhline(y=brier_upper_limit, color="r", linestyle="--", label="Upper Limit")
            if brier_lower_limit:
                ax.axhline(y=brier_lower_limit, color="r", linestyle="--", label="Lower Limit")
            ax.set_ylabel("Brier Score")
            ax.set_xlabel("Month")
            ax.legend()
            ax.grid(True)
            # Check for alerts
            if brier_lower_limit or brier_upper_limit:
                alerts = df[(df["Brier"] < brier_lower_limit) | (df["Brier"] > brier_upper_limit)]
                if not alerts.empty:
                    print("Brier Score Alerts:")
                    print(alerts)

        fig.tight_layout()
        return fig

    def add_model_for_plotting(self, model, X, y, label):
        self.models_to_plot.append({"model": model, "X": X, "y": y, "label": label})

    def plot_roc_curves(self, spec=""):
        fig, ax = plt.subplots()
        for model_info in self.models_to_plot:
            RocCurveDisplay.from_estimator(
                model_info["model"], model_info["X"], model_info["y"], name=model_info["label"], ax=ax
            )
        ax.set_title(f"ROC Curves {spec}")
        return fig, ax

    def plot_calibration_curves(self, spec=""):
        fig, ax = plt.subplots()
        for model_info in self.models_to_plot:
            CalibrationDisplay.from_estimator(
                model_info["model"], model_info["X"], model_info["y"], n_bins=20, name=model_info["label"], ax=ax
            )
        ax.set_title(f"Calibration Curves {spec}")
        return fig, ax

    def plot_combined_curves(self, spec=""):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for model_info in self.models_to_plot:
            RocCurveDisplay.from_estimator(
                model_info["model"], model_info["X"], model_info["y"], name=model_info["label"], ax=ax1
            )
            CalibrationDisplay.from_estimator(
                model_info["model"], model_info["X"], model_info["y"], n_bins=20, name=model_info["label"], ax=ax2
            )
        ax1.set_title(f"ROC Curves {spec}")
        ax2.set_title(f"Calibration Curves {spec}")
        fig.tight_layout()
        return fig
