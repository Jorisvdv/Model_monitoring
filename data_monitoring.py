# %% [markdown]
# # Nanny ML

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import nannyml as nml
import numpy as np

# %%
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# %%
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
# Settings for pdf export notebook
import plotly.io as pio

rederer = "svg"
pio.renderers.default = f"plotly_mimetype+{rederer}"
static_renderer = pio.renderers[rederer]
static_renderer.scale = 3


# %%
from load_data import CATEGORICAL_FEATURES_RENAMED, RENAME_LABELS_DICT, X_test, X_train, y_test, y_train

# %%
figure_folder = Path.cwd() / "figures"
figure_folder.mkdir(exist_ok=True)
result_folder = Path.cwd() / "results"
result_folder.mkdir(exist_ok=True)

# Y_EMC = pd.read_parquet(cached_data_folder / "Y_emc_day_02.parquet")
# Y_Treant = pd.read_parquet(cached_data_folder / "Y_treant_day_02.parquet")
# X_EMC = pd.read_parquet(cached_data_folder / "X_emc_day_02.parquet")
# X_Treant = pd.read_parquet(cached_data_folder / "X_treant_day_02.parquet")

# %%
SAVE_RESULTS = False

# %% [markdown]
# Data drift detection

# %%
X_train = X_train.rename(columns=RENAME_LABELS_DICT)
X_test = X_test.rename(columns=RENAME_LABELS_DICT)

# %%
uni_calc = nml.UnivariateDriftCalculator(
    column_names=X_train.drop(columns="Hospital").columns,
    treat_as_categorical=CATEGORICAL_FEATURES_RENAMED,
    timestamp_column_name="admission_start_time",
    continuous_methods=["kolmogorov_smirnov"],
    categorical_methods=["jensen_shannon"],
    # continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
    # categorical_methods=["chi2", "jensen_shannon"],
    chunk_period="M",
)

# %%
uni_calc.fit(X_train)

# %%
results = uni_calc.calculate(X_test)

# %%
results_EMC = uni_calc.calculate(X_test[X_test.Hospital == "EMC"])
results_Treant = uni_calc.calculate(X_test[X_test.Hospital == "Treant"])

# %%
test_results_df = results.to_df()
# test_results_df = test_results_df.loc[:-1] # Drop last (non complete) chunk

if SAVE_RESULTS:
    test_results_df.to_csv(result_folder / "Univariate_data_drift.csv")

    test_results_df.to_excel(result_folder / "Univariate_data_drift.xlsx")

# %%
analysis_period = test_results_df.loc[:, ("chunk", "chunk", "period")] == "analysis"


# %%
figure = results.plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_all.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_all.svg", scale=4)

# %%
figure = results.filter(metrics=X_test.drop(columns=["Hospital"]).columns.to_list()).plot(kind="distribution")
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_all.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_all.svg", scale=4)
figure.show()

# %%
figure_EMC = results_EMC.filter(metrics=X_test.drop(columns=["Hospital"]).columns.to_list()).plot(kind="distribution")
figure_Treant = results_Treant.filter(metrics=X_test.drop(columns=["Hospital"]).columns.to_list()).plot(
    kind="distribution"
)

if SAVE_RESULTS:
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_distribution_EMC.png", scale=4)
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_distribution_EMC.svg", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_distribution_Treant.png", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_distribution_Treant.svg", scale=4)

# %%
figure = results.filter(column_names=["Respiratory rate"]).plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_resp.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_resp.svg", scale=4)

# %%
figure_EMC = results_EMC.filter(column_names=["Respiratory rate"]).plot(kind="drift")
figure_Treant = results_Treant.filter(column_names=["Respiratory rate"]).plot(kind="drift")

if SAVE_RESULTS:
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_drift_resp_EMC.png", scale=4)
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_drift_resp_EMC.svg", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_drift_resp_Treant.png", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_drift_resp_Treant.svg", scale=4)
figure.show()

# %%
figure = results.filter(column_names=["Respiratory rate"]).plot(kind="distribution")
figure_EMC = results_EMC.filter(metrics=X_test.drop(columns=["Hospital"]).columns.to_list()).plot(kind="distribution")
figure_Treant = results_Treant.filter(metrics=X_test.drop(columns=["Hospital"]).columns.to_list()).plot(
    kind="distribution"
)

if SAVE_RESULTS:
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_distribution_EMC.png", scale=4)
    figure_EMC.write_image(f"{figure_folder}/Combined_Univariate_distribution_EMC.svg", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_distribution_Treant.png", scale=4)
    figure_Treant.write_image(f"{figure_folder}/Combined_Univariate_distribution_Treant.svg", scale=4)
figure.show()

# %%


# %%
figure = results.filter(column_names=["Saturation"]).plot(kind="distribution")
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_sat.png", scale=4, width=1000)
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_sat.svg", scale=4, width=1000)
figure.show()

# %% [markdown]
# Record Treant
# Admission: 2021-03-01 12:08:00
# SpO2: 955
#

# %%
alert_slice = pd.IndexSlice[:, :, "alert"]
alert_active = test_results_df.loc[analysis_period, alert_slice].any()
alert_column_names = alert_active[alert_active].index.get_level_values(0).unique().tolist()
print(alert_column_names)
figure = results.filter(column_names=alert_column_names).plot(kind="drift")
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_alert.png", scale=4, width=1000)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_alert.svg", scale=4, width=1000)
figure.show()

# %%
alert_slice = pd.IndexSlice[:, :, "alert"]
alert_active = test_results_df.loc[analysis_period, alert_slice].any()
alert_column_names = alert_active[alert_active].index.get_level_values(0).unique().tolist()
print(alert_column_names)

figure = results.filter(column_names=alert_column_names).plot(kind="distribution")
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_alert.png", scale=4, width=1000)
    figure.write_image(f"{figure_folder}/Combined_Univariate_distribution_alert.svg", scale=4, width=1000)
figure.show()

# %%
figure = results.filter(column_names=results.continuous_column_names, methods=["kolmogorov_smirnov"]).plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_conintuous.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_conintuous.svg", scale=4)

# %%
continuous_alert_slice = pd.IndexSlice[results.continuous_column_names, "kolmogorov_smirnov", "alert"]
continuous_alert_active = test_results_df.loc[analysis_period, continuous_alert_slice].any()
continuous_alert_column_names = (
    continuous_alert_active[continuous_alert_active].index.get_level_values(0).unique().tolist()
)
print(continuous_alert_column_names)
figure = results.filter(column_names=continuous_alert_column_names, methods=["kolmogorov_smirnov"]).plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_conintuous_alert.png", scale=4, width=1500)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_conintuous_alert.svg", scale=4, width=1500)

# %%
figure = results.filter(column_names=continuous_alert_column_names, methods=["kolmogorov_smirnov"]).plot(
    kind="distribution"
)
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_conintuous_alert_dist.png", scale=4, width=1000)
    figure.write_image(f"{figure_folder}/Combined_Univariate_conintuous_alert_dist.svg", scale=4, width=1000)

# %%
figure = results.filter(column_names=results.categorical_column_names, methods=["jensen_shannon"]).plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_categorical.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_categorical.svg", scale=4)

# %%
categorical_alert_slice = pd.IndexSlice[results.categorical_column_names, "jensen_shannon", "alert"]
categorical_alert_active = test_results_df.loc[:, categorical_alert_slice].any()
categorical_alert_column_names = (
    categorical_alert_active[categorical_alert_active].index.get_level_values(0).unique().tolist()
)
print(categorical_alert_column_names)
figure = results.filter(column_names=categorical_alert_column_names, methods=["jensen_shannon"]).plot(kind="drift")
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_categorical_alerts.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Univariate_drift_categorical_alerts.svg", scale=4)

# %%
figure = results.filter(column_names=categorical_alert_column_names, methods=["jensen_shannon"]).plot(
    kind="distribution"
)
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Univariate_categorical_alerts_dist.png", scale=4, width=1500)
    figure.write_image(f"{figure_folder}/Combined_Univariate_categorical_alerts_dist.svg", scale=4, width=1500)

# %%
fig, (ax1) = plt.subplots(1, 1)

X_train[X_train.Hospital == "Treant"].set_index("admission_start_time")["Respiratory rate"].resample(
    MonthEnd(n=2)
).mean().plot(ax=ax1, label="Treant")
# ax1.title("Mean respiratory rate Treant over time")

# ax1.xlabel("Year")
# fig.ylabel("Respiratory rate")

X_train[X_train.Hospital == "EMC"].set_index("admission_start_time")["Respiratory rate"].resample(
    MonthEnd(n=2)
).mean().plot(ax=ax1, label="EMC")
# ax2.title("Mean respiratory rate EMC over time")

# Adding title and axis labels
# ax1.set_title("Mean Respiratory Rate")
ax1.set_xlabel("Year")
ax1.set_ylabel("Mean respiratory Rate")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Display the legend
ax1.legend()
if SAVE_RESULTS:
    plt.savefig(figure_folder / "Mean_res.png")
    plt.savefig(figure_folder / "Mean_res.svg")
plt.show()

# %%
alert_months = test_results_df.loc[analysis_period].loc[
    test_results_df.loc[analysis_period, alert_slice].any(axis=1),
    [pd.IndexSlice["chunk", "chunk", "key"]],
]
alerts = pd.concat([alert_months, test_results_df.loc[alert_months.index, alert_slice]], axis=1)
alerts.columns = ["_".join([c[0], c[-1]]) for c in alerts.columns.to_flat_index()]
alerts = alerts.rename(columns={"chunk_key": "month"})
alerts = alerts.set_index("month")
print(alerts)

# %% [markdown]
# # Multivariate

# %%
X_train["Operation type"] = X_train["Operation type"].cat.codes
X_test["Operation type"] = X_test["Operation type"].cat.codes


# %%
def transform_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype("int")
    return df


bool_columns = [
    "Minimal invasive surgery",
    "ICU admission after surgery",
    "Sex",
    "Emergency admission",
    "IV antibiotics between surgery and prediction time",
    "Radiologic intervention between surgery and prediction time",
    "Reoperation between initial surgery and prediction time",
]

X_train = transform_to_int(X_train, bool_columns)
X_test = transform_to_int(X_test, bool_columns)

# %%
X_train.columns

# %%
multi_calc = nml.DataReconstructionDriftCalculator(
    column_names=X_train.drop(columns=["Hospital", "admission_start_time"]).columns,
    timestamp_column_name="admission_start_time",
    chunk_period="M",
    imputer_categorical=SimpleImputer(strategy="most_frequent", missing_values=pd.NA),
    # imputer_continuous=SimpleImputer(strategy="median"),
)
multi_calc.fit(X_train)

# %%
multi_results = multi_calc.calculate(X_test)

# %%
multi_test_results_df = multi_results.to_df()
if SAVE_RESULTS:
    multi_test_results_df.to_csv(result_folder / "Multivariate_data_drift.csv")
    multi_test_results_df.to_excel(result_folder / "Multivariate_data_drift.xlsx")

# %%
figure = multi_results.plot()
figure.show()
if SAVE_RESULTS:
    figure.write_image(f"{figure_folder}/Combined_Multivariate_drift.png", scale=4)
    figure.write_image(f"{figure_folder}/Combined_Multivariate_drift.svg", scale=4)

# %%
X_test_cleaned = X_test.copy()
X_test_cleaned.loc[X_test["Saturation"] > 100, "Saturation"] = pd.NA
X_test_cleaned.loc[X_test["Heart rate"] > 200, "Heart rate"] = pd.NA
X_test_cleaned = X_test_cleaned.loc[~(X_test_cleaned.index == X_test_cleaned["Length of stay before surgery"].idxmax())]
multi_results_cleaned = multi_calc.calculate(X_test_cleaned)
figure = multi_results_cleaned.plot()
figure.show()

# %%
multi_calc_exclude_resp = nml.DataReconstructionDriftCalculator(
    column_names=X_train.drop(columns=["Hospital", "admission_start_time", "Respiratory rate"]).columns,
    timestamp_column_name="admission_start_time",
    chunk_period="M",
    imputer_categorical=SimpleImputer(strategy="most_frequent", missing_values=pd.NA),
    # imputer_continuous=SimpleImputer(strategy="median"),
)
multi_calc_exclude_resp.fit(X_train)
multi_results_cleaned_exclude_resp = multi_calc_exclude_resp.calculate(X_test_cleaned)
figure = multi_results_cleaned_exclude_resp.plot()
figure.show()

# %%
multi_test_results_df

# %%
multi_alert_months = multi_test_results_df.loc[analysis_period].loc[
    multi_test_results_df.loc[analysis_period, pd.IndexSlice[:, "alert"]].any(axis=1),
    [pd.IndexSlice["chunk", "key"]],
]
multi_alerts = pd.concat(
    [
        multi_alert_months,
        multi_test_results_df.loc[multi_alert_months.index, pd.IndexSlice[:, "alert"]],
    ],
    axis=1,
)
multi_alerts.columns = ["_".join([c[0], c[-1]]) for c in multi_alerts.columns.to_flat_index()]
multi_alerts = multi_alerts.rename(columns={"chunk_key": "month"})
print(multi_alerts)

# %%
corresponding_uni_alerts = test_results_df.loc[
    multi_alerts.index,
    pd.IndexSlice[:, :, "alert"],
]
corresponding_uni_alerts.columns = ["_".join(c) for c in corresponding_uni_alerts.columns.to_flat_index()]

multi_alerts_months = multi_test_results_df.loc[multi_alerts.index, pd.IndexSlice["chunk", "key"]]
# multi_alerts_months.name = "_".join(multi_alerts_months.name)
multi_alerts_months.name = "month"
corresponding_uni_alerts = pd.concat([multi_alerts_months, corresponding_uni_alerts], axis=1)
# corresponding_uni_alerts["month"] = pd.to_datetime(corresponding_uni_alerts["month"])
corresponding_uni_alerts = corresponding_uni_alerts.set_index("month")
with pd.option_context("display.max_columns", None):
    print(corresponding_uni_alerts)

# %%
