# %% [markdown]
# # Model results

import itertools
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np

# %%
import pandas as pd

# import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from scipy.stats import ks_2samp, norm, ttest_ind
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# %%
# # Configure logging
# from utils.logger import create_logger

# logger = create_logger(filename="model")

# %%
figure_folder = Path.cwd() / "figures"
result_folder = Path.cwd().parent / "chirurgie-desire" / "results"

from load_data import CATEGORICAL_FEATURES_RENAMED, RENAME_LABELS_DICT, X_test, X_train, date_split, y_test, y_train

# %%
mlflow.set_tracking_uri(f"sqlite:///{result_folder}/mlruns.db")


# %%
class ModelEvaluator:
    def __init__(self, models):
        """
        Initializes the ModelEvaluator with a dictionary of models.

        Parameters:
        models (dict): A dictionary with model names as keys and trained model objects as values.
        """
        self.models = models

    @staticmethod
    def calculate_metrics(y_true, y_prob, threshold=0.5):
        """
        Calculates all specified metrics based on true labels, predicted probabilities, and threshold.
        """
        # AUC and APRC
        auc = roc_auc_score(y_true, y_prob)
        aprc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)

        # Obtain binary predictions based on the threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity, Specificity, PPV, and NPV calculations
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        return {
            "AUC": auc,
            "APRC": aprc,
            "Brier Score": brier,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
            "Threshold": threshold,
        }

    @staticmethod
    def find_optimal_threshold(y_true, y_prob):
        """
        Finds an optimal threshold that maximizes a specified criterion (e.g., Youden's Index).
        """
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # Youden’s Index = Sensitivity + Specificity - 1
        optimal_threshold = 0.5  # default value
        best_youden_index = 0

        for threshold in thresholds:
            metrics = ModelEvaluator.calculate_metrics(y_true, y_prob, threshold)
            youden_index = metrics["Sensitivity"] + metrics["Specificity"] - 1
            if youden_index > best_youden_index:
                best_youden_index = youden_index
                optimal_threshold = threshold

        return optimal_threshold

    def evaluate(self, X_test, y_test, threshold=0.5, optimize_threshold=False):
        """
        Evaluates multiple models on specified metrics and outputs a DataFrame with the results.
        """
        results = []
        for model_name, model in self.models.items():
            y_prob = model.predict_proba(X_test)[:, 1]

            # Optimize threshold if required
            if optimize_threshold:
                threshold = self.find_optimal_threshold(y_test, y_prob)

            metrics = self.calculate_metrics(y_test, y_prob, threshold)
            metrics["Model"] = model_name
            results.append(metrics)

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        return results_df


# %%
class ModelPlotter:
    def __init__(self):
        self.roc_displays = []
        self.calibration_displays = []
        self.auc = {}
        self.brier = {}

    def add_model(self, model, X_test, y_test, label):
        y_prob = model.predict_proba(X_test)[:, 1]
        self.auc[label] = roc_auc_score(y_true=y_test, y_score=y_prob)
        self.brier[label] = brier_score_loss(y_true=y_test, y_proba=y_prob)
        label_with_brier = f"{label} (Brier = {self.brier[label]:.3f})"

        # Store ROC display object
        roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test, name=label)
        self.roc_displays.append(roc_display)

        # Store Calibration display object
        calibration_display = CalibrationDisplay.from_estimator(model, X_test, y_test, name=label_with_brier, n_bins=10)
        self.calibration_displays.append(calibration_display)

    def plot_all_roc_curves(self, spec: str = ""):
        fig, ax = plt.subplots()
        for display in self.roc_displays:
            display.plot(ax=ax)
        plt.title(f"ROC Curves {spec}")
        plt.show()
        return fig, ax

    def plot_all_calibration_curves(self, spec: str = ""):
        fig, ax = plt.subplots()
        for display in self.calibration_displays:
            display.plot(ax=ax)
        plt.title(f"Calibration Curve{spec}")
        plt.show()
        return fig, ax

    def plot_combined_curves(self, spec: str = ""):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot ROC curves in the first subplot
        for display in self.roc_displays:
            display.plot(ax=ax1)
        ax1.set_title("A: ROC Curves")
        ax1.set_aspect("equal")  # Make the ROC plot square

        # Plot Calibration curves in the second subplot
        for display in self.calibration_displays:
            display.plot(ax=ax2)
        ax2.set_title("B: Calibration Curves")
        ax2.set_aspect("equal")  # Make the Calibration plot square

        # Set consistent x and y limits for both subplots to match shape
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        fig.suptitle(f"ROC and Calibration Curves {spec}")
        plt.show()

        return fig  # Return fig to allow saving


# %%
rf_best_model: Pipeline = mlflow.sklearn.load_model("runs:/469494ffd6e34d5c8c2eb763a4f30056/model")
xgb_best_model: Pipeline = mlflow.sklearn.load_model("runs:/979a6af2420a44258c53b84c03b8ba6e/model")
logreg_best_model: Pipeline = mlflow.sklearn.load_model("runs:/2f23d4b1dce9486495a81cc5e84b0b95/model")

# %%
models = {
    # "Random Forest": rf_best_model,
    "XGBoost": xgb_best_model,
    # "Logistic Regression": logreg_best_model,
}

# %%
# Create an instance of ModelEvaluator
evaluator = ModelEvaluator(models)

# Evaluate models with a fixed threshold of 0.5
results_df = evaluator.evaluate(X_test, y_test, threshold=0.9, optimize_threshold=False)

# Display results
print(results_df)

# %%
# Step 3: Plot ROC and Calibration Curves for the best models
plotter = ModelPlotter()

# Add the best model from each nested cross-validation
# plotter.add_model(rf_best_model, X_test, y_test, label="Random Forest")
plotter.add_model(xgb_best_model, X_test, y_test, label="XGBoost")
# plotter.add_model(logreg_best_model, X_test, y_test, label="Logistic Regression")

roc_curve, _ = plotter.plot_all_roc_curves(spec="for temporal validation set")
# roc_curve.figure.savefig(figure_folder / "ROC_temp_val.png")
# roc_curve.figure.savefig(figure_folder / "ROC_temp_val.svg")
calibration_curve, ax_cal_val = plotter.plot_all_calibration_curves(spec="")
# calibration_curve.figure.savefig(figure_folder / "Cal_temp_val.png")
# calibration_curve.figure.savefig(figure_folder / "Cal_temp_val.svg")
combined_plots = plotter.plot_combined_curves(spec="for temporal validation set")
# combined_plots.figure.savefig(figure_folder / "Combined_temp_val.png")
# combined_plots.figure.savefig(figure_folder / "Combined_temp_val.svg")

# %%
X_test_EMC = X_test[X_test["hospital"] == "EMC"].drop(columns=["hospital"])
X_test_Treant = X_test[X_test["hospital"] == "Treant"].drop(columns=["hospital"])

y_test_EMC = X_test_EMC.merge(y_test, how="left", left_index=True, right_index=True)["safe_discharge"]
y_test_Treant = X_test_Treant.merge(y_test, how="left", left_index=True, right_index=True)["safe_discharge"]

# %%
# Step 3: Plot ROC and Calibration Curves for the best models
plotter = ModelPlotter()

# Add the best model from each nested cross-validation
# plotter.add_model(rf_best_model, X_test_EMC, y_test_EMC, label="Random Forest (EMC)")
# plotter.add_model(rf_best_model, X_test_Treant, y_test_Treant, label="Random Forest (Treant)")
plotter.add_model(xgb_best_model, X_test_EMC, y_test_EMC, label="XGBoost (EMC)")
plotter.add_model(xgb_best_model, X_test_Treant, y_test_Treant, label="XGBoost (Treant)")
# plotter.add_model(logreg_best_model, X_test_EMC, y_test_EMC, label="LR (EMC)")
# plotter.add_model(logreg_best_model, X_test_Treant, y_test_Treant, label="LR (Treant)")

roc_hospital, _ = plotter.plot_all_roc_curves(spec="Temporal validation set, per hospital")
roc_hospital.figure.savefig(figure_folder / "ROC_temp_hospital.png")
roc_hospital.figure.savefig(figure_folder / "ROC_temp_hospital.svg")

calibration_hospital, ax_cal_hospital = plotter.plot_all_calibration_curves(
    spec="Temporal validation set, per hospital"
)
calibration_hospital.figure.savefig(figure_folder / "Cal_temp_hospital.png")
calibration_hospital.figure.savefig(figure_folder / "Cal_temp_hospital.svg")

combined_plots_hospital = plotter.plot_combined_curves(spec="Temporal validation set, per hospital")
combined_plots_hospital.figure.savefig(figure_folder / "Combined_temp_hospital.png")
combined_plots_hospital.figure.savefig(figure_folder / "Combined_temp_hospital.svg")

# %%
# # Combined Train and Test Data with Cutoff Date
# combined_data = pd.concat([train_data, test_data])
# cutoff_date = "2020-01"


# %%
class MonthlyPerformanceCalculator:
    def __init__(self, models, target_column, date_column, rolling_window=3):
        self.models = models
        self.target_column = target_column
        self.date_column = date_column
        self.rolling_window = rolling_window
        self.initialize_performance_dict()

    def initialize_performance_dict(self):
        """Initialize the monthly performance dictionary dynamically for each model."""
        self.monthly_performance = {"month": []}
        for model_name in self.models.keys():
            self.monthly_performance[f"auc_{model_name}"] = []
            self.monthly_performance[f"brier_{model_name}"] = []

    def calculate_monthly_performance(self, data: pd.DataFrame):
        """Calculate AUC and Brier score for each model on a monthly basis."""
        self.initialize_performance_dict()
        data = data.set_index(pd.to_datetime(data[self.date_column]))

        # Resample data monthly with rolling_window offset
        resampled_data = data.resample(pd.offsets.MonthEnd(self.rolling_window))

        for month, group in resampled_data:
            X_month = group.drop(columns=[self.target_column, self.date_column])
            y_month = group[self.target_column]

            # Skip if only one outcome (mostly in last month)
            if y_month.nunique() == 1:
                continue

            self.monthly_performance["month"].append(month)

            for model_name, model in self.models.items():
                y_pred = model.predict_proba(X_month)[:, 1]
                self.monthly_performance[f"auc_{model_name}"].append(roc_auc_score(y_month, y_pred))
                self.monthly_performance[f"brier_{model_name}"].append(brier_score_loss(y_month, y_pred))

        self.monthly_performance_df = pd.DataFrame(self.monthly_performance)

    def calculate_95_ci(self, series):
        """Calculate 95% confidence interval based on normal approximation."""
        if len(series) == 0:
            return np.nan, np.nan, np.nan
        mean = series.mean()
        std_error = series.std() / np.sqrt(len(series))
        margin = std_error * norm.ppf(0.975)  # 95% CI
        return mean, mean - margin, mean + margin

    def export_overall_statistics(self):
        """Calculate and export overall mean and 95% CI for AUC and Brier Score across the entire period."""
        rows = []

        for model_name in self.models.keys():
            # Calculate statistics for AUC
            auc_series = self.monthly_performance_df[f"auc_{model_name}"]
            auc_mean, auc_ci_lower, auc_ci_upper = self.calculate_95_ci(auc_series)

            # Calculate statistics for Brier Score
            brier_series = self.monthly_performance_df[f"brier_{model_name}"]
            brier_mean, brier_ci_lower, brier_ci_upper = self.calculate_95_ci(brier_series)

            # Append the results as a row in the rows list
            rows.append(
                {
                    "Model": model_name,
                    "AUC Mean": auc_mean,
                    "AUC 95% CI Lower": auc_ci_lower,
                    "AUC 95% CI Upper": auc_ci_upper,
                    "Brier Mean": brier_mean,
                    "Brier 95% CI Lower": brier_ci_lower,
                    "Brier 95% CI Upper": brier_ci_upper,
                }
            )

        # Convert the rows list to a DataFrame with one row per model
        overall_statistics_df = pd.DataFrame(rows)
        return overall_statistics_df


# %%
class PerformancePlotter:
    def __init__(self, performance_df):
        self.performance_df = performance_df

    def plot_monthly_performance(
        self,
        title: str,
        cutoff_date=None,
        plot_rolling_avg=False,
        plot_std_dev=False,
        only_rolling_avg=False,
        auc_upper_limit: float | None = None,
        auc_lower_limit: float | None = None,
        brier_upper_limit: float | None = None,
        brier_lower_limit: float | None = None,
        plot_auc=True,
        plot_brier=True,
        remove_last_month=False,
        mark_exceeding_limits=False,
        figure=None,
    ):
        """Plot AUROC and/or Brier Score for each model with optional rolling averages, standard deviations, limit lines, and highlighting of limit exceedances."""
        if not plot_auc and not plot_brier:
            raise ValueError("At least one of plot_auc or plot_brier must be True")

        # Remove the last month if requested
        if remove_last_month and len(self.performance_df) > 1:
            self.performance_df = self.performance_df.iloc[:-1]

        fig, ax1 = plt.subplots(figsize=(12, 8))

        auc_lines = []  # Lines for AUROC-related legend
        auc_labels = []  # Labels for AUROC-related legend
        brier_lines = []  # Lines for Brier-related legend
        brier_labels = []  # Labels for Brier-related legend

        color_cycle = itertools.cycle(
            plt.rcParams["axes.prop_cycle"].by_key()["color"]
        )  # Color cycle to assign distinct colors

        if plot_auc:
            for column in self.performance_df.columns:
                if column.startswith("auc_"):
                    model_name = column.split("_", 1)[1]
                    color = next(color_cycle)

                    # Plot rolling average and optionally standard deviation
                    if plot_rolling_avg:
                        rolling_mean = self.performance_df[column].rolling(window=3, min_periods=1).mean()

                        (rolling_mean_line,) = ax1.plot(
                            self.performance_df["month"],
                            rolling_mean,
                            linestyle="--",
                            color=color,
                            label=f"{model_name} Rolling AUROC Mean",
                        )
                        auc_lines.append(rolling_mean_line)
                        auc_labels.append(f"{model_name} Rolling AUROC Mean")

                        if plot_std_dev:
                            rolling_std = self.performance_df[column].rolling(window=3, min_periods=1).std()
                            ax1.fill_between(
                                self.performance_df["month"],
                                rolling_mean - rolling_std,
                                rolling_mean + rolling_std,
                                color=color,
                                alpha=0.2,
                            )

                    # Plot AUROC values if not only rolling average
                    if not only_rolling_avg:
                        (line,) = ax1.plot(
                            self.performance_df["month"],
                            self.performance_df[column],
                            marker="o",
                            linestyle="-",
                            color=color,
                            label=f"{model_name} AUROC",
                        )
                        auc_lines.append(line)
                        auc_labels.append(f"{model_name} AUROC")

                        # Highlight values exceeding AUROC limits if required
                        if mark_exceeding_limits:
                            exceeding_idx = self.performance_df[
                                (self.performance_df[column] > auc_upper_limit)
                                | (self.performance_df[column] < auc_lower_limit)
                            ].index

                            ax1.plot(
                                self.performance_df.loc[exceeding_idx, "month"],
                                self.performance_df.loc[exceeding_idx, column],
                                "D",
                                color="red",
                                markersize=8,
                                label=f"{model_name} AUROC Exceeding Limit" if column == "auc_" + model_name else None,
                            )

            ax1.set_ylabel("AUROC")

            # Add AUROC limit lines if provided
            if auc_upper_limit is not None:
                auc_upper_limit_line = ax1.axhline(
                    auc_upper_limit,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label="AUROC Limit",
                )
                auc_lines.append(auc_upper_limit_line)
                auc_labels.append("AUROC Limit")
            if auc_lower_limit is not None:
                auc_lower_limit_line = ax1.axhline(
                    auc_lower_limit,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label="AUROC Limit",
                )
                if "AUROC Limit" not in auc_labels:
                    auc_lines.append(auc_lower_limit_line)

        if plot_brier:
            ax2 = ax1 if not plot_auc else ax1.twinx()
            for column in self.performance_df.columns:
                if column.startswith("brier_"):
                    model_name = column.split("_", 1)[1]
                    color = next(color_cycle)

                    # Plot rolling average and optionally standard deviation
                    if plot_rolling_avg:
                        rolling_mean = self.performance_df[column].rolling(window=3, min_periods=1).mean()

                        (rolling_mean_line,) = ax2.plot(
                            self.performance_df["month"],
                            rolling_mean,
                            linestyle="--",
                            color=color,
                            label=f"{model_name} Rolling Brier Mean",
                        )
                        brier_lines.append(rolling_mean_line)
                        brier_labels.append(f"{model_name} Rolling Brier Mean")

                        if plot_std_dev:
                            rolling_std = self.performance_df[column].rolling(window=3, min_periods=1).std()
                            ax2.fill_between(
                                self.performance_df["month"],
                                rolling_mean - rolling_std,
                                rolling_mean + rolling_std,
                                color=color,
                                alpha=0.2,
                            )

                    # Plot Brier Score values if not only rolling average
                    if not only_rolling_avg:
                        (line,) = ax2.plot(
                            self.performance_df["month"],
                            self.performance_df[column],
                            marker="s",
                            linestyle=":",
                            color=color,
                            label=f"{model_name} Brier Score",
                        )
                        brier_lines.append(line)
                        brier_labels.append(f"{model_name} Brier Score")

                        # Highlight values exceeding Brier limits if required
                        if mark_exceeding_limits:
                            exceeding_idx = self.performance_df[
                                (self.performance_df[column] > brier_upper_limit)
                                | (self.performance_df[column] < brier_lower_limit)
                            ].index

                            ax2.plot(
                                self.performance_df.loc[exceeding_idx, "month"],
                                self.performance_df.loc[exceeding_idx, column],
                                "D",
                                color="red",
                                markersize=8,
                                label=(
                                    f"{model_name} Brier Exceeding Limit" if column == "brier_" + model_name else None
                                ),
                            )

            ax2.set_ylabel("Brier Score")

            # Add Brier limit lines if provided
            if brier_upper_limit is not None:
                brier_upper_limit_line = ax2.axhline(
                    brier_upper_limit,
                    color="blue",
                    linestyle="--",
                    alpha=0.5,
                    label="Brier Limit",
                )
                brier_lines.append(brier_upper_limit_line)
                brier_labels.append("Brier Limit")
            if brier_lower_limit is not None:
                brier_lower_limit_line = ax2.axhline(
                    brier_lower_limit,
                    color="blue",
                    linestyle="--",
                    alpha=0.5,
                    label="Brier Limit",
                )
                if "Brier Limit" not in brier_labels:
                    brier_lines.append(brier_lower_limit_line)

        # Add cutoff date line if provided
        if cutoff_date:
            cutoff_line = plt.axvline(
                pd.to_datetime(cutoff_date),
                color="cyan",
                linestyle="dashdot",
                label=f"Cutoff Date ({cutoff_date})",
            )
            auc_lines.append(cutoff_line)
            auc_labels.append(f"Cutoff Date ({cutoff_date})")

        # Set separate legends for AUROC and Brier scores
        if plot_auc:
            ax1.legend(auc_lines, auc_labels, loc="upper left")
        if plot_brier:
            ax2.legend(brier_lines, brier_labels, loc="upper right")

        plt.title(title)
        plt.xlabel("Month")
        fig.autofmt_xdate()
        plt.tight_layout()

        # Return the figure and axes to allow saving
        return fig, ax1, ax2 if plot_brier else fig, ax1


# %%
validation_score_mean = pd.read_excel(result_folder / "Validation_score.xlsx", index_col=0)
validation_score_mean = validation_score_mean["XGBoost"].to_dict()

# %%
validation_score_mean

# %%
# Define mean and standard deviation for AUC to set performance limits
mean_AUC = validation_score_mean["mean_ROC"]  # validation_score_mean["mean_ROC"]
AUC_SD = validation_score_mean["std_ROC"]
auc_upper_limit = mean_AUC + (3 * AUC_SD)
auc_lower_limit = mean_AUC - (3 * AUC_SD)

# Define mean and standard deviation for brier to set performance limits
mean_brier = validation_score_mean["mean_Brier"]  # validation_score_mean["mean_Brier"]
brier_SD = validation_score_mean["std_Brier"]
brier_upper_limit = mean_brier + (3 * brier_SD)
brier_lower_limit = mean_brier - (3 * brier_SD)


# %%
test_data = pd.merge(X_test, y_test, left_index=True, right_index=True)
# %%
# Create an instance of MonthlyPerformanceCalculator and perform calculations
models = {
    # "Random Forest": rf_best_model,
    "XGBoost": xgb_best_model,
    # "Logistic Regression": logreg_best_model,
}

calculator_XGB_1_month = MonthlyPerformanceCalculator(
    models=models,
    target_column="safe_discharge",
    date_column="admission_start_time",
    rolling_window=1,
)
calculator_XGB_1_month.calculate_monthly_performance(test_data)
performance_df_XGB_1_month = calculator_XGB_1_month.monthly_performance_df

# calculator_3_months = MonthlyPerformanceCalculator(
#     models=models,
#     target_column="safe_discharge",
#     date_column="admission_start_time",
#     rolling_window=1,
# )
# calculator_3_months.calculate_monthly_performance(test_data)
# performance_df_3_months = calculator_3_months.monthly_performance_df

# %%
# Create an instance of PerformancePlotter and plot results
plotter_1_month_XGB = PerformancePlotter(performance_df=performance_df_XGB_1_month)
# Example usage with limits, plotting AUC and Brier, removing the last month, and saving the plot
fig = plotter_1_month_XGB.plot_monthly_performance(
    title="Model AUC over time",
    auc_upper_limit=None,
    auc_lower_limit=auc_lower_limit,
    brier_upper_limit=brier_upper_limit,
    brier_lower_limit=brier_lower_limit,
    plot_auc=True,
    plot_brier=False,
    # plot_rolling_avg=True,
    # plot_std_dev=True,
    remove_last_month=False,
)


# %%
fig = plt.figure(figsize=(11, 4))
gs = fig.add_gridspec(nrows=11, ncols=12)

ax_calibration_curve = fig.add_subplot(gs[:8, :4])


display: CalibrationDisplay = CalibrationDisplay.from_estimator(
    xgb_best_model,
    X_test,
    y_test,
    n_bins=20,
    name="XGBoost",
    ax=ax_calibration_curve,
)

ax_calibration_curve.grid()
ax_calibration_curve.set_title("A: Calibration plot")

# Add histogram
ax_hist = fig.add_subplot(gs[-2:, :4], sharex=ax_calibration_curve)
ax_hist.hist(
    display.y_prob,
    range=(0, 1),
    bins=20,
)
ax_hist.set(title="Histogram", xlabel="Predicted probability", ylabel="Count")
ax_calibration_curve.set_xlim(0, 1)
ax_calibration_curve.set_ylim(0, 1)
ax_calibration_curve.set_xlabel("Predicted Probability")
ax_calibration_curve.set_ylabel("Fraction of Positives")
# ax_calibration_curve.legend()
ax_calibration_curve.grid(True)
ax_calibration_curve.legend(loc="upper left")
ax_calibration_curve.spines["top"].set_visible(False)
ax_calibration_curve.spines["right"].set_visible(False)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)
# Set aspect ratio to be equal.
# ax_calibration_curve.set_aspect("equal", adjustable="box")
ax_perf = fig.add_subplot(gs[:, 5:])
ax_perf.spines["top"].set_visible(False)
ax_perf.spines["right"].set_visible(False)
plot_df_XGB_1_month = performance_df_XGB_1_month.set_index("month")
plot_df_XGB_1_month["auc_XGBoost"].plot(ax=ax_perf, label="XGBoost")
ax_perf.axhline(
    auc_lower_limit,
    color="red",
    linestyle="--",
    alpha=0.5,
    label="AUROC Limit",
)
ax_perf.set_title("B: Monthly model performance")
ax_perf.set_xlabel("Month")
ax_perf.set_ylabel("AUROC")
ax_perf.legend()
fig.savefig(figure_folder / "Temporal_val_figure.png")
fig.savefig(figure_folder / "Temporal_val_figure.svg")
plt.show()

# axs[1].set_ylabel("AUROC")

# %%
fig = plt.figure(figsize=(11, 4))
gs = fig.add_gridspec(nrows=11, ncols=12)

ax_calibration_curve = fig.add_subplot(gs[:8, :4])

display_EMC: CalibrationDisplay = CalibrationDisplay.from_estimator(
    xgb_best_model,
    X_test_EMC,
    y_test_EMC,
    n_bins=20,
    name="XGBoost (EMC)",
    ax=ax_calibration_curve,
)
display_Treant: CalibrationDisplay = CalibrationDisplay.from_estimator(
    xgb_best_model,
    X_test_Treant,
    y_test_Treant,
    n_bins=20,
    name="XGBoost (Treant)",
    ax=ax_calibration_curve,
)

ax_calibration_curve.grid()
ax_calibration_curve.set_title("A: Calibration plot")

# Add histogram
ax_hist = fig.add_subplot(gs[-2:, :4], sharex=ax_calibration_curve)
ax_hist.hist(
    display_EMC.y_prob,
    range=(0, 1),
    bins=20,
    label="XGBoost (EMC)",
    alpha=0.7,
)
ax_hist.hist(
    display_Treant.y_prob,
    range=(0, 1),
    bins=20,
    label="XGBoost (Treant)",
    alpha=0.7,
)
ax_hist.set(title="Histogram", xlabel="Predicted probability", ylabel="Count")
ax_calibration_curve.set_xlim(0, 1)
ax_calibration_curve.set_ylim(0, 1)
ax_calibration_curve.set_xlabel("Predicted Probability")
ax_calibration_curve.set_ylabel("Fraction of Positives")
# ax_calibration_curve.legend()
ax_calibration_curve.grid(True)
ax_calibration_curve.legend(loc="upper left")
ax_calibration_curve.spines["top"].set_visible(False)
ax_calibration_curve.spines["right"].set_visible(False)
ax_hist.spines["top"].set_visible(False)
ax_hist.spines["right"].set_visible(False)
# Set aspect ratio to be equal.
# ax_calibration_curve.set_aspect("equal", adjustable="box")
ax_perf = fig.add_subplot(gs[:, 5:])
ax_perf.spines["top"].set_visible(False)
ax_perf.spines["right"].set_visible(False)
calculator_XGB_EMC = MonthlyPerformanceCalculator(
    models=models,
    target_column="safe_discharge",
    date_column="admission_start_time",
    rolling_window=1,
)
calculator_XGB_EMC.calculate_monthly_performance(test_data.loc[test_data["hospital"] == "EMC"])
performance_df_XGB_EMC = calculator_XGB_EMC.monthly_performance_df
plot_df_XGB_EMC = performance_df_XGB_EMC.set_index("month")
plot_df_XGB_EMC["auc_XGBoost"].plot(ax=ax_perf, label="XGBoost (EMC)")
calculator_XGB_Treant = MonthlyPerformanceCalculator(
    models=models,
    target_column="safe_discharge",
    date_column="admission_start_time",
    rolling_window=1,
)
calculator_XGB_Treant.calculate_monthly_performance(test_data.loc[test_data["hospital"] == "Treant"])
performance_df_XGB_Treant = calculator_XGB_Treant.monthly_performance_df
plot_df_XGB_Treant = performance_df_XGB_Treant.set_index("month")
plot_df_XGB_Treant["auc_XGBoost"].plot(ax=ax_perf, label="XGBoost (Treant)")
# ax_perf.axhline(
#     auc_lower_limit,
#     color="red",
#     linestyle="--",
#     alpha=0.5,
#     label="AUROC Limit",
# )
ax_perf.set_title("B: Monthly model performance")
ax_perf.set_xlabel("Month")
ax_perf.set_ylabel("AUROC")
ax_perf.legend()
fig.savefig(figure_folder / "Temporal_val_figure_hospital.png")
fig.savefig(figure_folder / "Temporal_val_figure_hospital.svg")
plt.show()

# axs[1].set_ylabel("AUROC")

# %%
# Create an instance of MonthlyPerformanceCalculator and perform calculations
model = {
    "XGBoost": xgb_best_model,
}

calculator_XGB_1_month = MonthlyPerformanceCalculator(
    models=model,
    target_column="safe_discharge",
    date_column="admission_start_time",
    rolling_window=1,
)
calculator_XGB_1_month.calculate_monthly_performance(test_data)
performance_df_XGB_1_month = calculator_XGB_1_month.monthly_performance_df

# Create an instance of PerformancePlotter and plot results
plotter_1_month_XGB = PerformancePlotter(performance_df=performance_df_XGB_1_month)
# Example usage with limits, plotting AUC and Brier, removing the last month, and saving the plot
fig = plotter_1_month_XGB.plot_monthly_performance(
    title="B: Monthly performance during temporal validation",
    auc_upper_limit=None,
    auc_lower_limit=auc_lower_limit,
    brier_upper_limit=brier_upper_limit,
    brier_lower_limit=brier_lower_limit,
    plot_auc=True,
    plot_brier=False,
    # plot_rolling_avg=True,
    # plot_std_dev=True,
    remove_last_month=False,
)
# fig.savefig(figure_folder / f"{'Model AUC over time'.replace(' ', '_')}.png")
# fig.savefig(figure_folder / f"{'Model AUC over time'.replace(' ', '_')}.svg")

# fig, ax1, ax2 = plotter_1_month_XGB.plot_monthly_performance(
#     title="Model Brier score over time",
#     auc_upper_limit=None,
#     auc_lower_limit=auc_lower_limit,
#     brier_upper_limit=brier_upper_limit,
#     brier_lower_limit=brier_lower_limit,
#     plot_auc=False,
#     plot_brier=True,
#     # plot_rolling_avg=True,
#     # plot_std_dev=True,
#     remove_last_month=False,
# )

# %%
print(
    "AUC performance limit",
    *performance_df_XGB_1_month.loc[
        performance_df_XGB_1_month["auc_XGBoost"] < auc_lower_limit, "month"
    ].dt.date.to_list(),
    sep="\n",
)
print(
    "Brier performance limit",
    *performance_df_XGB_1_month.loc[
        performance_df_XGB_1_month["brier_XGBoost"] > brier_upper_limit, "month"
    ].dt.date.to_list(),
    sep="\n",
)

# %%
# %%
