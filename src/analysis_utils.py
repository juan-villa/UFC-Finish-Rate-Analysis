from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

CURRENT_DIVISIONS = [
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
]

DEFAULT_HEATMAP_ORDER = [
    "Women's Strawweight",
    "Women's Bantamweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
]

DEFAULT_OUTCOME_ORDER = [
    "Women's Strawweight",
    "Women's Flyweight",
    "Women's Bantamweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Light Heavyweight",
    "Heavyweight",
]

NUMERIC_COLUMNS = [
    "Round",
    "Title",
    "Fighter_A_KD",
    "Fighter_B_KD",
    "Fighter_A_STR",
    "Fighter_B_STR",
    "Fighter_A_TD",
    "Fighter_B_TD",
    "Fighter_A_SUB",
    "Fighter_B_SUB",
]


@dataclass
class ModelBundle:
    analysis_df: pd.DataFrame
    y: pd.Series
    X: pd.DataFrame
    logit_model: Any
    probit_model: Any
    logit_prob: np.ndarray
    probit_prob: np.ndarray


OUTCOME_MAP = {
    "KO/TKO": "KO",
    "SUB": "SUB",
    "U-DEC": "UDEC",
    "S-DEC": "SDEC",
    "M-DEC": "MDEC",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_raw_path() -> Path:
    return get_project_root() / "data" / "raw" / "ufc_event_data.csv"


def default_processed_path() -> Path:
    return get_project_root() / "data" / "processed" / "ufc_event_data.csv"


def load_fight_data(path: str | Path | None = None) -> pd.DataFrame:
    candidate = Path(path) if path else None

    if candidate is None:
        raw_path = default_raw_path()
        processed_path = default_processed_path()
        if raw_path.exists():
            candidate = raw_path
        elif processed_path.exists():
            candidate = processed_path
        else:
            raise FileNotFoundError(
                "no dataset found. run python -m src.scraping first or place csv at data/processed/ufc_event_data.csv"
            )

    if not candidate.exists():
        raise FileNotFoundError(f"dataset not found at {candidate}")

    logger.info("loading dataset from %s", candidate)
    return pd.read_csv(candidate)


def save_processed_data(dataframe: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path else default_processed_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    logger.info("saved processed dataset to %s", path)
    return path


def classify_outcome(victory_result: Any) -> str:
    if pd.isna(victory_result):
        return "DRAW"
    return OUTCOME_MAP.get(str(victory_result).strip(), "DRAW")


def prepare_analysis_frame(
    dataframe: pd.DataFrame,
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-01",
    allowed_divisions: list[str] | None = None,
) -> pd.DataFrame:
    allowed = allowed_divisions if allowed_divisions is not None else CURRENT_DIVISIONS

    df = dataframe.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Year"] = df["Date"].dt.year

    df = df[df["Weight_Class"].isin(allowed)]
    df = df[(df["Date"] > pd.Timestamp(start_date)) & (df["Date"] < pd.Timestamp(end_date))].copy()

    df["Outcome"] = df["Victory_Result"].apply(classify_outcome)
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def compute_yearly_finish_stats(dataframe: pd.DataFrame) -> pd.DataFrame:
    finish_mask = dataframe["Outcome"].isin(["KO", "SUB"])

    yearly_stats = dataframe.groupby("Year").agg(Total=("Outcome", "count"))
    yearly_stats["Finishes"] = dataframe.loc[finish_mask].groupby("Year").size()
    yearly_stats["Finishes"] = yearly_stats["Finishes"].fillna(0)

    yearly_stats["Finish_Rate"] = yearly_stats["Finishes"] / yearly_stats["Total"]
    yearly_stats["Finish_Rate_Change"] = yearly_stats["Finish_Rate"].diff()
    yearly_stats["Finish_Rate_Change_Pct"] = yearly_stats["Finish_Rate"].pct_change() * 100
    return yearly_stats


def compute_division_finish_rates(
    dataframe: pd.DataFrame,
    weight_order: list[str] | None = None,
) -> pd.DataFrame:
    finish_mask = dataframe["Outcome"].isin(["KO", "SUB"])

    division_stats = (
        dataframe.groupby(["Year", "Weight_Class"])
        .agg(Total=("Outcome", "count"))
        .reset_index()
    )

    finishes = (
        dataframe[finish_mask]
        .groupby(["Year", "Weight_Class"])
        .size()
        .reset_index(name="Finishes")
    )

    division_stats = division_stats.merge(finishes, on=["Year", "Weight_Class"], how="left")
    division_stats["Finishes"] = division_stats["Finishes"].fillna(0)
    division_stats["Finish_Rate"] = division_stats["Finishes"] / division_stats["Total"]

    heatmap_data = division_stats.pivot(index="Weight_Class", columns="Year", values="Finish_Rate")
    if weight_order:
        heatmap_data = heatmap_data.reindex(weight_order)

    return heatmap_data


def compute_outcome_mix_by_division(
    dataframe: pd.DataFrame,
    weight_order: list[str] | None = None,
    include_outcomes: list[str] | None = None,
) -> pd.DataFrame:
    outcomes = include_outcomes or ["KO", "SUB", "UDEC", "SDEC", "MDEC"]
    filtered_df = dataframe[dataframe["Outcome"].isin(outcomes)].copy()

    finish_counts = (
        filtered_df.groupby(["Weight_Class", "Outcome"])
        .size()
        .reset_index(name="Count")
    )
    finish_pivot = (
        finish_counts.pivot(index="Weight_Class", columns="Outcome", values="Count")
        .fillna(0)
    )
    finish_prop = finish_pivot.div(finish_pivot.sum(axis=1), axis=0)

    if weight_order:
        finish_prop = finish_prop.reindex(weight_order)

    return finish_prop


def build_model_matrix(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    model_df = dataframe[
        [
            "Outcome",
            "Year",
            "Round",
            "Title",
            "Weight_Class",
            "Fighter_A_KD",
            "Fighter_B_KD",
            "Fighter_A_STR",
            "Fighter_B_STR",
            "Fighter_A_TD",
            "Fighter_B_TD",
            "Fighter_A_SUB",
            "Fighter_B_SUB",
        ]
    ].copy()

    model_df["Finish"] = model_df["Outcome"].isin(["KO", "SUB"]).astype(int)
    model_df["Total_KD"] = model_df["Fighter_A_KD"] + model_df["Fighter_B_KD"]
    model_df["Total_SIG_STR"] = model_df["Fighter_A_STR"] + model_df["Fighter_B_STR"]
    model_df["Total_TD"] = model_df["Fighter_A_TD"] + model_df["Fighter_B_TD"]
    model_df["Total_SUB_ATT"] = model_df["Fighter_A_SUB"] + model_df["Fighter_B_SUB"]
    model_df["SIG_STR_DIFF_ABS"] = (model_df["Fighter_A_STR"] - model_df["Fighter_B_STR"]).abs()
    model_df["Year_c"] = model_df["Year"] - model_df["Year"].mean()

    base_features = [
        "Year_c",
        "Round",
        "Title",
        "Total_KD",
        "Total_SIG_STR",
        "Total_TD",
        "Total_SUB_ATT",
        "SIG_STR_DIFF_ABS",
    ]

    x_base = model_df[base_features]
    weight_dummies = pd.get_dummies(model_df["Weight_Class"], prefix="WC", drop_first=True, dtype=int)
    x_matrix = pd.concat([x_base, weight_dummies], axis=1)
    x_matrix = sm.add_constant(x_matrix, has_constant="add")

    y_target = model_df["Finish"]
    analysis_df = pd.concat([y_target.rename("Finish"), x_matrix], axis=1).dropna()
    y = analysis_df["Finish"].astype(int)
    x = analysis_df.drop(columns=["Finish"]).astype(float)

    return analysis_df, y, x


def fit_finish_models(dataframe: pd.DataFrame) -> ModelBundle:
    analysis_df, y, x = build_model_matrix(dataframe)

    logit_model = sm.Logit(y, x).fit(disp=False)
    probit_model = sm.Probit(y, x).fit(disp=False)

    logit_prob = logit_model.predict(x).to_numpy()
    probit_prob = probit_model.predict(x).to_numpy()

    return ModelBundle(
        analysis_df=analysis_df,
        y=y,
        X=x,
        logit_model=logit_model,
        probit_model=probit_model,
        logit_prob=logit_prob,
        probit_prob=probit_prob,
    )


def build_calibration_table(bundle: ModelBundle, bins: int = 10) -> pd.DataFrame:
    viz_df = bundle.analysis_df.copy()
    viz_df["pred_logit"] = bundle.logit_prob
    viz_df["pred_probit"] = bundle.probit_prob

    viz_df["pred_bin"] = pd.qcut(viz_df["pred_logit"], q=bins, duplicates="drop")
    calibration = (
        viz_df.groupby("pred_bin", observed=True)
        .agg(
            Observed_Finish=("Finish", "mean"),
            Logit_Pred=("pred_logit", "mean"),
            Probit_Pred=("pred_probit", "mean"),
            N=("Finish", "size"),
        )
        .reset_index(drop=True)
    )
    return calibration


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.unique(np.r_[1.0, y_score, 0.0])[::-1]
    tpr: list[float] = []
    fpr: list[float] = []

    positives = (y_true == 1).sum()
    negatives = (y_true == 0).sum()

    for threshold in thresholds:
        pred = (y_score >= threshold).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr.append(tp / positives if positives > 0 else 0.0)
        fpr.append(fp / negatives if negatives > 0 else 0.0)

    fpr_array = np.array(fpr)
    tpr_array = np.array(tpr)
    order = np.argsort(fpr_array)
    return fpr_array[order], tpr_array[order]


def compute_auc_curves(bundle: ModelBundle) -> dict[str, Any]:
    y_true = bundle.y.to_numpy()

    fpr_logit, tpr_logit = roc_points(y_true, bundle.logit_prob)
    fpr_probit, tpr_probit = roc_points(y_true, bundle.probit_prob)

    auc_logit = np.trapezoid(tpr_logit, fpr_logit)
    auc_probit = np.trapezoid(tpr_probit, fpr_probit)

    return {
        "y_true": y_true,
        "fpr_logit": fpr_logit,
        "tpr_logit": tpr_logit,
        "fpr_probit": fpr_probit,
        "tpr_probit": tpr_probit,
        "auc_logit": auc_logit,
        "auc_probit": auc_probit,
    }


def confusion_matrix_df(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    y_pred = (y_prob >= threshold).astype(int)
    return (
        pd.crosstab(
            pd.Series(y_true, name="Actual"),
            pd.Series(y_pred, name="Predicted"),
            dropna=False,
        )
        .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        .astype(int)
    )


def compute_confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())

    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }
