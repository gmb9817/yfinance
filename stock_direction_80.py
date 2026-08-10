from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class ResearchConfig:
    ticker: str
    start: str
    target_accuracy: float
    test_days: int
    oof_days: int
    fold_days: int
    min_train_days: int
    gap_days: int
    min_calibration_signals: int
    min_test_signals: int
    min_coverage: float
    min_side_signals: int
    confidence_level: float
    refit_days: int
    mode: str
    output_dir: str
    random_state: int


KOREAN_PANEL = [
    "005930.KS",
    "000660.KS",
    "035420.KS",
    "035720.KS",
    "051910.KS",
    "006400.KS",
    "207940.KS",
    "005380.KS",
    "000270.KS",
    "068270.KS",
    "105560.KS",
    "055550.KS",
]

US_PANEL = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "AVGO",
    "AMD",
    "JPM",
    "XOM",
    "UNH",
    "COST",
]

KOREAN_CONTEXT = {
    "kospi": ("^KS11", 0),
    "kosdaq": ("^KQ11", 0),
    "spy": ("SPY", 1),
    "qqq": ("QQQ", 1),
    "sox": ("^SOX", 1),
    "vix": ("^VIX", 1),
    "usdkrw": ("KRW=X", 1),
}

US_CONTEXT = {
    "spy": ("SPY", 0),
    "qqq": ("QQQ", 0),
    "iwm": ("IWM", 0),
    "sox": ("^SOX", 0),
    "vix": ("^VIX", 0),
    "tnx": ("^TNX", 0),
}

META_COLUMNS = {
    "date",
    "ticker",
    "target",
    "future_return",
    "future_abs_return",
    "close_price",
    "is_latest",
}


def unique_ordered(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def is_korean_ticker(ticker: str) -> bool:
    upper = ticker.upper()
    return upper.endswith(".KS") or upper.endswith(".KQ")


def get_universe(ticker: str, mode: str) -> tuple[list[str], dict[str, tuple[str, int]]]:
    if is_korean_ticker(ticker):
        base = KOREAN_PANEL if mode == "full" else KOREAN_PANEL[:7]
        return unique_ordered([ticker] + base), KOREAN_CONTEXT
    base = US_PANEL if mode == "full" else US_PANEL[:7]
    return unique_ordered([ticker] + base), US_CONTEXT


def import_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. pip install -U yfinance를 실행하세요.") from exc
    return yf


def normalize_history(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close", "Volume"}
        selected = None
        for level in range(frame.columns.nlevels):
            values = set(map(str, frame.columns.get_level_values(level)))
            if fields.issubset(values):
                other_levels = [i for i in range(frame.columns.nlevels) if i != level]
                if other_levels:
                    try:
                        selected = frame.xs(ticker, axis=1, level=other_levels[0], drop_level=True)
                    except Exception:
                        selected = None
                if selected is None:
                    columns = {}
                    for field in fields:
                        matches = [col for col in frame.columns if str(col[level]) == field]
                        if matches:
                            columns[field] = frame[matches[0]]
                    selected = pd.DataFrame(columns, index=frame.index)
                frame = selected
                break
    renamed = {str(column).strip().title(): column for column in frame.columns}
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(name in renamed for name in required):
        return pd.DataFrame()
    frame = frame[[renamed[name] for name in required]].copy()
    frame.columns = required
    index = pd.to_datetime(frame.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.sort_index()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    frame = frame[frame["Close"] > 0]
    return frame


def download_history(ticker: str, start: str) -> pd.DataFrame:
    yf = import_yfinance()
    errors: list[str] = []
    attempts = [
        {"interval": "1d", "auto_adjust": True, "actions": False, "repair": True},
        {"interval": "1d", "auto_adjust": True, "actions": False},
    ]
    for settings in attempts:
        try:
            history = yf.Ticker(ticker).history(start=start, **settings)
            normalized = normalize_history(history, ticker)
            if not normalized.empty:
                return normalized
        except Exception as exc:
            errors.append(str(exc))
    try:
        history = yf.download(
            ticker,
            start=start,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            repair=True,
        )
        normalized = normalize_history(history, ticker)
        if not normalized.empty:
            return normalized
    except Exception as exc:
        errors.append(str(exc))
    message = " | ".join(errors[-3:])
    raise RuntimeError(f"{ticker} 데이터를 받지 못했습니다: {message}")


def download_universe(tickers: list[str], start: str) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    histories: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for position, ticker in enumerate(tickers, start=1):
        print(f"[{position:02d}/{len(tickers):02d}] {ticker} 다운로드")
        try:
            histories[ticker] = download_history(ticker, start)
        except Exception as exc:
            failures[ticker] = str(exc)
    return histories, failures


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))

    def calculate(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        return float(np.dot(values, centered) / denominator)

    return series.rolling(window, min_periods=window).apply(calculate, raw=True)


def rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = safe_divide(gain, loss)
    return 100 - 100 / (1 + relative_strength)


def build_local_features(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = history.copy()
    open_price = frame["Open"]
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    volume = frame["Volume"].fillna(0)
    previous_close = close.shift(1)
    log_close = np.log(close)
    log_return = log_close.diff()
    result = pd.DataFrame(index=frame.index)
    result["ticker"] = ticker
    result["close_price"] = close
    result["gap_return"] = safe_divide(open_price, previous_close) - 1
    result["intraday_return"] = safe_divide(close, open_price) - 1
    result["range_ratio"] = safe_divide(high - low, close)
    result["body_ratio"] = safe_divide((close - open_price).abs(), high - low)
    result["upper_wick_ratio"] = safe_divide(high - pd.concat([open_price, close], axis=1).max(axis=1), high - low)
    result["lower_wick_ratio"] = safe_divide(pd.concat([open_price, close], axis=1).min(axis=1) - low, high - low)
    result["close_location"] = safe_divide(close - low, high - low)
    for horizon in [1, 2, 3, 5, 10, 20, 60]:
        result[f"return_{horizon}"] = close.pct_change(horizon)
    for lag in range(1, 11):
        result[f"log_return_lag_{lag}"] = log_return.shift(lag - 1)
    for window in [5, 10, 20, 60]:
        rolling_returns = log_return.rolling(window, min_periods=window)
        result[f"return_mean_{window}"] = rolling_returns.mean()
        result[f"volatility_{window}"] = rolling_returns.std()
        result[f"return_skew_{window}"] = rolling_returns.skew()
        result[f"return_kurt_{window}"] = rolling_returns.kurt()
        result[f"price_sma_ratio_{window}"] = safe_divide(close, close.rolling(window, min_periods=window).mean()) - 1
        result[f"trend_slope_{window}"] = rolling_slope(log_close, window)
        result[f"drawdown_{window}"] = safe_divide(close, close.rolling(window, min_periods=window).max()) - 1
    for window in [5, 12, 26, 60]:
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        result[f"price_ema_ratio_{window}"] = safe_divide(close, ema) - 1
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = safe_divide(ema_12 - ema_26, close)
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd"] = macd
    result["macd_signal"] = macd_signal
    result["macd_histogram"] = macd - macd_signal
    for window in [7, 14, 28]:
        result[f"rsi_{window}"] = rsi(close, window) / 100
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    result["atr_14"] = safe_divide(true_range.rolling(14, min_periods=14).mean(), close)
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std()
    result["bollinger_z_20"] = safe_divide(close - middle, deviation)
    rolling_low = low.rolling(14, min_periods=14).min()
    rolling_high = high.rolling(14, min_periods=14).max()
    stochastic = safe_divide(close - rolling_low, rolling_high - rolling_low)
    result["stochastic_14"] = stochastic
    result["stochastic_signal_3"] = stochastic.rolling(3, min_periods=3).mean()
    log_volume = np.log1p(volume.clip(lower=0))
    result["volume_change_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    result["volume_change_5"] = volume.pct_change(5).replace([np.inf, -np.inf], np.nan)
    for window in [5, 20, 60]:
        volume_mean = log_volume.rolling(window, min_periods=window).mean()
        volume_std = log_volume.rolling(window, min_periods=window).std()
        result[f"volume_z_{window}"] = safe_divide(log_volume - volume_mean, volume_std)
        result[f"volume_ratio_{window}"] = safe_divide(volume, volume.rolling(window, min_periods=window).mean()) - 1
    signed_volume = np.sign(close.diff()).fillna(0) * volume
    obv = signed_volume.cumsum()
    result["obv_slope_10"] = rolling_slope(obv, 10)
    result["obv_slope_20"] = rolling_slope(obv, 20)
    weekday = result.index.dayofweek
    month = result.index.month
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 5)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 5)
    result["month_sin"] = np.sin(2 * np.pi * month / 12)
    result["month_cos"] = np.cos(2 * np.pi * month / 12)
    future_return = close.shift(-1) / close - 1
    result["future_return"] = future_return
    result["future_abs_return"] = future_return.abs()
    result["target"] = np.where(future_return.notna(), (future_return > 0).astype(float), np.nan)
    result["is_latest"] = False
    valid_latest = result.drop(columns=["target", "future_return", "future_abs_return"]).replace([np.inf, -np.inf], np.nan)
    if not valid_latest.empty:
        result.loc[result.index[-1], "is_latest"] = True
    return result


def build_context_features(history: pd.DataFrame, prefix: str, lag: int) -> pd.DataFrame:
    close = history["Close"]
    log_return = np.log(close).diff()
    result = pd.DataFrame(index=history.index)
    for horizon in [1, 2, 5, 10, 20, 60]:
        result[f"context_{prefix}_return_{horizon}"] = close.pct_change(horizon)
    for window in [5, 20, 60]:
        result[f"context_{prefix}_volatility_{window}"] = log_return.rolling(window, min_periods=window).std()
        result[f"context_{prefix}_trend_{window}"] = safe_divide(close, close.rolling(window, min_periods=window).mean()) - 1
        result[f"context_{prefix}_drawdown_{window}"] = safe_divide(close, close.rolling(window, min_periods=window).max()) - 1
    if lag > 0:
        result = result.shift(lag)
    return result


def build_panel_dataset(
    target_ticker: str,
    panel_histories: dict[str, pd.DataFrame],
    context_histories: dict[str, tuple[pd.DataFrame, int]],
) -> tuple[pd.DataFrame, list[str]]:
    local_frames = [build_local_features(history, ticker) for ticker, history in panel_histories.items()]
    if not local_frames:
        raise RuntimeError("학습 가능한 패널 데이터가 없습니다.")
    context_frames = []
    for prefix, (history, lag) in context_histories.items():
        context_frames.append(build_context_features(history, prefix, lag))
    context_matrix = pd.concat(context_frames, axis=1).sort_index() if context_frames else pd.DataFrame()
    merged_frames = []
    for local in local_frames:
        merged = local.copy()
        if not context_matrix.empty:
            aligned = context_matrix.reindex(merged.index).ffill(limit=5)
            merged = merged.join(aligned, how="left")
        merged_frames.append(merged)
    panel = pd.concat(merged_frames, axis=0)
    panel.index.name = "date"
    panel = panel.reset_index()
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    for column in ["return_1", "return_5", "return_20", "volume_z_20", "volatility_20"]:
        if column not in panel.columns:
            continue
        grouped = panel.groupby("date", sort=False)[column]
        panel[f"cross_rank_{column}"] = grouped.rank(pct=True, method="average")
        panel[f"cross_mean_{column}"] = grouped.transform("mean")
        panel[f"cross_std_{column}"] = grouped.transform("std")
        panel[f"relative_{column}"] = panel[column] - panel[f"cross_mean_{column}"]
    panel["market_breadth"] = panel.groupby("date", sort=False)["return_1"].transform(lambda values: (values > 0).mean())
    panel["market_dispersion"] = panel.groupby("date", sort=False)["return_1"].transform("std")
    ticker_dummies = pd.get_dummies(panel["ticker"], prefix="ticker_id", dtype=float)
    panel = pd.concat([panel, ticker_dummies], axis=1)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    feature_columns = [
        column
        for column in panel.columns
        if column not in META_COLUMNS and column != "date" and pd.api.types.is_numeric_dtype(panel[column])
    ]
    feature_availability = panel[feature_columns].notna().mean(axis=1)
    panel = panel[(feature_availability >= 0.70) | panel["is_latest"]].reset_index(drop=True)
    target_rows = panel[panel["ticker"] == target_ticker]
    if target_rows.empty:
        raise RuntimeError(f"{target_ticker}가 최종 패널에 없습니다.")
    return panel, feature_columns


def make_walk_forward_splits(
    dates: pd.DatetimeIndex,
    oof_days: int,
    fold_days: int,
    min_train_days: int,
    gap_days: int,
    max_train_days: int | None,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    start_position = max(min_train_days + gap_days, len(unique_dates) - oof_days)
    splits: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    validation_start = start_position
    while validation_start < len(unique_dates):
        validation_end = min(validation_start + fold_days, len(unique_dates))
        training_end = validation_start - gap_days
        if training_end < min_train_days:
            validation_start = validation_end
            continue
        training_start = 0 if max_train_days is None else max(0, training_end - max_train_days)
        training_dates = unique_dates[training_start:training_end]
        validation_dates = unique_dates[validation_start:validation_end]
        if len(training_dates) >= min_train_days and len(validation_dates) > 0:
            splits.append((training_dates, validation_dates))
        validation_start = validation_end
    return splits


def build_models(mode: str, random_state: int) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "logistic": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.08,
                        max_iter=4000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500 if mode == "full" else 280,
                        max_depth=9,
                        min_samples_leaf=18,
                        max_features=0.55,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.035,
                        max_iter=350 if mode == "full" else 220,
                        max_leaf_nodes=15,
                        max_depth=5,
                        min_samples_leaf=28,
                        l2_regularization=8.0,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    if mode == "full" and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=520,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=18,
                        subsample=0.82,
                        colsample_bytree=0.68,
                        reg_alpha=0.6,
                        reg_lambda=12.0,
                        gamma=0.05,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    return models


def fit_predict_models(
    models: dict[str, Pipeline],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    predict_x: pd.DataFrame,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    fitted: dict[str, Pipeline] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        candidate = clone(model)
        candidate.fit(train_x, train_y)
        fitted[name] = candidate
        probabilities[name] = candidate.predict_proba(predict_x)[:, 1]
    return fitted, pd.DataFrame(probabilities, index=predict_x.index)


def generate_oof_predictions(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    development_dates: pd.DatetimeIndex,
    config: ResearchConfig,
    max_train_days: int | None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    splits = make_walk_forward_splits(
        development_dates,
        config.oof_days,
        config.fold_days,
        config.min_train_days,
        config.gap_days,
        max_train_days,
    )
    if len(splits) < 3:
        raise RuntimeError("워크포워드 폴드가 3개 미만입니다. 시작 날짜를 더 과거로 설정하세요.")
    models = build_models(config.mode, config.random_state)
    all_predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold_number, (training_dates, validation_dates) in enumerate(splits, start=1):
        train_mask = dataset["date"].isin(training_dates) & dataset["target"].notna()
        validation_mask = dataset["date"].isin(validation_dates) & dataset["target"].notna()
        train = dataset.loc[train_mask]
        validation = dataset.loc[validation_mask]
        if train.empty or validation.empty:
            continue
        train_y = train["target"].astype(int)
        if train_y.nunique() < 2:
            continue
        print(
            f"윈도우={max_train_days or 'expanding'} 폴드={fold_number}/{len(splits)} "
            f"학습={training_dates[0].date()}~{training_dates[-1].date()} "
            f"검증={validation_dates[0].date()}~{validation_dates[-1].date()}"
        )
        _, fold_probabilities = fit_predict_models(
            models,
            train[feature_columns],
            train_y,
            validation[feature_columns],
        )
        fold_output = validation[["date", "ticker", "target"]].copy()
        for name in fold_probabilities.columns:
            fold_output[f"prob_{name}"] = fold_probabilities[name].to_numpy()
        all_predictions.append(fold_output)
        target_fold = fold_output[fold_output["ticker"] == config.ticker]
        if not target_fold.empty:
            mean_probability = target_fold.filter(like="prob_").mean(axis=1)
            prediction = (mean_probability >= 0.5).astype(int)
            fold_records.append(
                {
                    "fold": fold_number,
                    "train_start": str(training_dates[0].date()),
                    "train_end": str(training_dates[-1].date()),
                    "validation_start": str(validation_dates[0].date()),
                    "validation_end": str(validation_dates[-1].date()),
                    "samples": int(len(target_fold)),
                    "accuracy": float(accuracy_score(target_fold["target"].astype(int), prediction)),
                }
            )
    if not all_predictions:
        raise RuntimeError("OOF 예측을 생성하지 못했습니다.")
    result = pd.concat(all_predictions, axis=0).sort_values(["date", "ticker"])
    return result, fold_records


def build_stack_features(probability_frame: pd.DataFrame) -> pd.DataFrame:
    probability_columns = [column for column in probability_frame.columns if column.startswith("prob_")]
    result = probability_frame[probability_columns].copy()
    clipped = result.clip(1e-5, 1 - 1e-5)
    for column in probability_columns:
        result[f"logit_{column}"] = np.log(clipped[column] / (1 - clipped[column]))
    result["probability_mean"] = probability_frame[probability_columns].mean(axis=1)
    result["probability_std"] = probability_frame[probability_columns].std(axis=1).fillna(0)
    result["probability_min"] = probability_frame[probability_columns].min(axis=1)
    result["probability_max"] = probability_frame[probability_columns].max(axis=1)
    result["raw_confidence"] = (result["probability_mean"] - 0.5).abs() * 2
    return result


def split_oof_dates(oof: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    dates = pd.DatetimeIndex(sorted(pd.unique(oof["date"])))
    if len(dates) < 180:
        raise RuntimeError("OOF 기간이 너무 짧습니다. oof_days를 늘리세요.")
    first_cut = max(1, int(len(dates) * 0.50))
    second_cut = max(first_cut + 1, int(len(dates) * 0.75))
    return dates[:first_cut], dates[first_cut:second_cut], dates[second_cut:]


def fit_stacker(oof: pd.DataFrame, stack_train_dates: pd.DatetimeIndex) -> Pipeline:
    stack_train = oof[oof["date"].isin(stack_train_dates)]
    x = build_stack_features(stack_train)
    y = stack_train["target"].astype(int)
    stacker = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.12,
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    stacker.fit(x, y)
    return stacker


def apply_stacker(stacker: Pipeline, probability_frame: pd.DataFrame) -> np.ndarray:
    return stacker.predict_proba(build_stack_features(probability_frame))[:, 1]


def selector_feature_names(feature_columns: list[str]) -> list[str]:
    preferred = [
        "return_1",
        "return_2",
        "return_5",
        "return_20",
        "volatility_5",
        "volatility_20",
        "volatility_60",
        "rsi_14",
        "bollinger_z_20",
        "atr_14",
        "volume_z_20",
        "market_breadth",
        "market_dispersion",
        "cross_rank_return_1",
        "cross_rank_return_5",
        "relative_return_1",
        "relative_return_5",
    ]
    context_candidates = [
        column
        for column in feature_columns
        if column.startswith("context_") and (column.endswith("return_1") or column.endswith("volatility_20") or column.endswith("trend_20"))
    ]
    return [column for column in preferred if column in feature_columns] + context_candidates[:10]


def build_selector_features(
    rows: pd.DataFrame,
    probability_frame: pd.DataFrame,
    stacked_probability: np.ndarray,
    selected_feature_columns: list[str],
) -> pd.DataFrame:
    base = build_stack_features(probability_frame).copy()
    base["stacked_probability"] = stacked_probability
    base["stacked_confidence"] = np.abs(stacked_probability - 0.5) * 2
    base["stacked_direction"] = (stacked_probability >= 0.5).astype(float)
    for column in selected_feature_columns:
        base[f"market_{column}"] = pd.to_numeric(rows[column], errors="coerce").to_numpy()
    return base


def fit_selector(
    dataset: pd.DataFrame,
    oof: pd.DataFrame,
    stacker: Pipeline,
    selector_train_dates: pd.DatetimeIndex,
    config: ResearchConfig,
    feature_columns: list[str],
) -> tuple[Any, list[str]]:
    target_oof = oof[(oof["ticker"] == config.ticker) & oof["date"].isin(selector_train_dates)].copy()
    if len(target_oof) < 60:
        raise RuntimeError("메타 선택기 학습 표본이 부족합니다.")
    probability_columns = [column for column in target_oof.columns if column.startswith("prob_")]
    probabilities = target_oof[probability_columns]
    stacked_probability = apply_stacker(stacker, probabilities)
    base_prediction = (stacked_probability >= 0.5).astype(int)
    correctness = (base_prediction == target_oof["target"].astype(int).to_numpy()).astype(int)
    selected_columns = selector_feature_names(feature_columns)
    lookup = dataset.set_index(["date", "ticker"])
    source_rows = lookup.loc[list(zip(target_oof["date"], target_oof["ticker"]))].reset_index()
    selector_x = build_selector_features(source_rows, probabilities.reset_index(drop=True), stacked_probability, selected_columns)
    if np.unique(correctness).size < 2:
        selector = DummyClassifier(strategy="prior")
        synthetic_x = pd.concat([selector_x.iloc[[0]], selector_x.iloc[[0]]], ignore_index=True)
        synthetic_y = np.array([0, 1])
        sample_weight = np.array([1e-6, 1.0]) if correctness[0] == 1 else np.array([1.0, 1e-6])
        selector.fit(synthetic_x, synthetic_y, sample_weight=sample_weight)
        return selector, selected_columns
    selector = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.05,
                    max_iter=3000,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    selector.fit(selector_x, correctness)
    return selector, selected_columns


def clopper_pearson_interval(successes: int, total: int, confidence_level: float) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    alpha = 1 - confidence_level
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def evaluate_selection(
    y_true: np.ndarray,
    direction_probability: np.ndarray,
    selection_score: np.ndarray,
    threshold: float,
    confidence_level: float,
) -> dict[str, Any]:
    selected = selection_score >= threshold
    predictions = (direction_probability >= 0.5).astype(int)
    selected_count = int(selected.sum())
    if selected_count == 0:
        return {
            "threshold": float(threshold),
            "signals": 0,
            "coverage": 0.0,
            "accuracy": math.nan,
            "lower": math.nan,
            "upper": math.nan,
            "up_signals": 0,
            "down_signals": 0,
            "errors": 0,
            "certification_p": math.nan,
        }
    selected_y = y_true[selected]
    selected_predictions = predictions[selected]
    successes = int((selected_y == selected_predictions).sum())
    errors = selected_count - successes
    lower, upper = clopper_pearson_interval(successes, selected_count, confidence_level)
    p_value = float(binomtest(errors, selected_count, p=0.2, alternative="less").pvalue)
    return {
        "threshold": float(threshold),
        "signals": selected_count,
        "coverage": float(selected_count / len(y_true)),
        "accuracy": float(successes / selected_count),
        "lower": lower,
        "upper": upper,
        "up_signals": int((selected_predictions == 1).sum()),
        "down_signals": int((selected_predictions == 0).sum()),
        "errors": errors,
        "certification_p": p_value,
    }


def choose_selection_threshold(
    y_true: np.ndarray,
    direction_probability: np.ndarray,
    selection_score: np.ndarray,
    config: ResearchConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    finite_scores = selection_score[np.isfinite(selection_score)]
    if len(finite_scores) == 0:
        raise RuntimeError("선택 점수가 모두 결측입니다.")
    quantiles = np.linspace(0.0, 0.98, 100)
    thresholds = np.unique(np.quantile(finite_scores, quantiles))[::-1]
    records = [
        evaluate_selection(y_true, direction_probability, selection_score, threshold, config.confidence_level)
        for threshold in thresholds
    ]
    table = pd.DataFrame(records)
    usable = table[
        (table["signals"] >= config.min_calibration_signals)
        & (table["coverage"] >= config.min_coverage)
        & (table["up_signals"] >= config.min_side_signals)
        & (table["down_signals"] >= config.min_side_signals)
    ].copy()
    multiple_testing_alpha = (1 - config.confidence_level) / max(1, len(table))
    certified = usable[
        (usable["accuracy"] >= config.target_accuracy)
        & (usable["certification_p"] <= multiple_testing_alpha)
    ]
    empirical = usable[usable["accuracy"] >= config.target_accuracy]
    if not certified.empty:
        chosen = certified.sort_values(["coverage", "lower", "accuracy"], ascending=False).iloc[0].to_dict()
        chosen["selection_status"] = "certified"
    elif not empirical.empty:
        chosen = empirical.sort_values(["coverage", "lower", "accuracy"], ascending=False).iloc[0].to_dict()
        chosen["selection_status"] = "empirical"
    elif not usable.empty:
        chosen = usable.sort_values(["accuracy", "lower", "coverage"], ascending=False).iloc[0].to_dict()
        chosen["selection_status"] = "target_not_found"
    else:
        chosen = table.sort_values(["accuracy", "signals"], ascending=False).iloc[0].to_dict()
        chosen["selection_status"] = "insufficient_signals"
    chosen["multiple_testing_alpha"] = multiple_testing_alpha
    return chosen, table


def prepare_research_configuration(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    development_dates: pd.DatetimeIndex,
    max_train_days: int | None,
    config: ResearchConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oof, fold_records = generate_oof_predictions(
        dataset,
        feature_columns,
        development_dates,
        config,
        max_train_days,
    )
    stack_dates, selector_dates, calibration_dates = split_oof_dates(oof)
    stacker = fit_stacker(oof, stack_dates)
    selector, selected_columns = fit_selector(
        dataset,
        oof,
        stacker,
        selector_dates,
        config,
        feature_columns,
    )
    calibration_oof = oof[(oof["ticker"] == config.ticker) & oof["date"].isin(calibration_dates)].copy()
    probability_columns = [column for column in calibration_oof.columns if column.startswith("prob_")]
    calibration_probabilities = calibration_oof[probability_columns].reset_index(drop=True)
    stacked_probability = apply_stacker(stacker, calibration_probabilities)
    lookup = dataset.set_index(["date", "ticker"])
    source_rows = lookup.loc[list(zip(calibration_oof["date"], calibration_oof["ticker"]))].reset_index()
    selector_x = build_selector_features(
        source_rows,
        calibration_probabilities,
        stacked_probability,
        selected_columns,
    )
    selection_score = selector.predict_proba(selector_x)[:, 1]
    y_true = calibration_oof["target"].astype(int).to_numpy()
    threshold_record, risk_coverage = choose_selection_threshold(
        y_true,
        stacked_probability,
        selection_score,
        config,
    )
    forced_prediction = (stacked_probability >= 0.5).astype(int)
    forced_accuracy = float(accuracy_score(y_true, forced_prediction))
    balanced_accuracy = float(balanced_accuracy_score(y_true, forced_prediction))
    auc = float(roc_auc_score(y_true, stacked_probability)) if len(np.unique(y_true)) > 1 else math.nan
    summary = {
        "max_train_days": max_train_days,
        "calibration_start": str(calibration_dates[0].date()),
        "calibration_end": str(calibration_dates[-1].date()),
        "calibration_samples": int(len(calibration_oof)),
        "forced_accuracy": forced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc,
        "brier": float(brier_score_loss(y_true, stacked_probability)),
        "log_loss": float(log_loss(y_true, stacked_probability, labels=[0, 1])),
        **threshold_record,
    }
    components = {
        "oof": oof,
        "fold_records": fold_records,
        "stacker": stacker,
        "selector": selector,
        "selector_columns": selected_columns,
        "risk_coverage": risk_coverage,
        "probability_columns": probability_columns,
    }
    return summary, components


def configuration_score(summary: dict[str, Any], config: ResearchConfig) -> tuple[float, float, float, float]:
    status = summary.get("selection_status")
    status_score = {"certified": 4.0, "empirical": 3.0, "target_not_found": 2.0, "insufficient_signals": 1.0}.get(status, 0.0)
    target_reached = 1.0 if float(summary.get("accuracy", 0.0)) >= config.target_accuracy else 0.0
    return (
        status_score,
        target_reached,
        float(summary.get("coverage", 0.0)),
        float(summary.get("lower", 0.0)) if np.isfinite(summary.get("lower", np.nan)) else 0.0,
    )


def predict_with_models(models: dict[str, Pipeline], x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"prob_{name}": model.predict_proba(x)[:, 1] for name, model in models.items()},
        index=x.index,
    )


def walk_forward_test(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    development_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    max_train_days: int | None,
    stacker: Pipeline,
    selector: Pipeline,
    selector_columns: list[str],
    selection_threshold: float,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    model_templates = build_models(config.mode, config.random_state)
    target_predictions: list[pd.DataFrame] = []
    last_fitted: dict[str, Pipeline] = {}
    for block_start in range(0, len(test_dates), config.refit_days):
        block_dates = test_dates[block_start : block_start + config.refit_days]
        first_test_date = block_dates[0]
        available_dates = pd.DatetimeIndex(sorted(pd.unique(dataset.loc[dataset["date"] < first_test_date, "date"])))
        if config.gap_days > 0:
            available_dates = available_dates[: -config.gap_days] if len(available_dates) > config.gap_days else pd.DatetimeIndex([])
        if max_train_days is not None and len(available_dates) > max_train_days:
            available_dates = available_dates[-max_train_days:]
        train = dataset[dataset["date"].isin(available_dates) & dataset["target"].notna()]
        test_block = dataset[
            dataset["date"].isin(block_dates)
            & dataset["target"].notna()
            & (dataset["ticker"] == config.ticker)
        ].copy()
        if train.empty or test_block.empty:
            continue
        print(
            f"테스트 재학습 {block_dates[0].date()}~{block_dates[-1].date()} "
            f"학습종료={available_dates[-1].date()}"
        )
        last_fitted, raw_probabilities = fit_predict_models(
            model_templates,
            train[feature_columns],
            train["target"].astype(int),
            test_block[feature_columns],
        )
        raw_probabilities.columns = [f"prob_{column}" for column in raw_probabilities.columns]
        stacked_probability = apply_stacker(stacker, raw_probabilities)
        selector_x = build_selector_features(
            test_block.reset_index(drop=True),
            raw_probabilities.reset_index(drop=True),
            stacked_probability,
            selector_columns,
        )
        selection_score = selector.predict_proba(selector_x)[:, 1]
        output = test_block[["date", "ticker", "target", "future_return", "close_price"]].copy()
        output["direction_probability"] = stacked_probability
        output["selection_score"] = selection_score
        output["prediction"] = (stacked_probability >= 0.5).astype(int)
        output["selected"] = selection_score >= selection_threshold
        output["correct"] = output["prediction"] == output["target"].astype(int)
        target_predictions.append(output)
    if not target_predictions:
        raise RuntimeError("테스트 예측을 생성하지 못했습니다.")
    return pd.concat(target_predictions).sort_values("date"), last_fitted


def summarize_test(predictions: pd.DataFrame, config: ResearchConfig) -> dict[str, Any]:
    y_true = predictions["target"].astype(int).to_numpy()
    y_pred = predictions["prediction"].astype(int).to_numpy()
    probability = predictions["direction_probability"].to_numpy()
    selected = predictions["selected"].to_numpy(dtype=bool)
    majority = int(np.mean(y_true) >= 0.5)
    majority_accuracy = float(np.mean(y_true == majority))
    summary: dict[str, Any] = {
        "test_start": str(predictions["date"].min().date()),
        "test_end": str(predictions["date"].max().date()),
        "test_samples": int(len(predictions)),
        "forced_accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "majority_baseline": majority_accuracy,
        "roc_auc": float(roc_auc_score(y_true, probability)) if len(np.unique(y_true)) > 1 else math.nan,
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "signals": int(selected.sum()),
        "coverage": float(selected.mean()),
    }
    if selected.any():
        selected_y = y_true[selected]
        selected_prediction = y_pred[selected]
        successes = int((selected_y == selected_prediction).sum())
        total = int(selected.sum())
        lower, upper = clopper_pearson_interval(successes, total, config.confidence_level)
        summary.update(
            {
                "selective_accuracy": float(successes / total),
                "selective_lower": lower,
                "selective_upper": upper,
                "up_signals": int((selected_prediction == 1).sum()),
                "down_signals": int((selected_prediction == 0).sum()),
            }
        )
    else:
        summary.update(
            {
                "selective_accuracy": math.nan,
                "selective_lower": math.nan,
                "selective_upper": math.nan,
                "up_signals": 0,
                "down_signals": 0,
            }
        )
    summary["passed_80"] = bool(
        summary["signals"] >= config.min_test_signals
        and summary["coverage"] >= config.min_coverage
        and np.isfinite(summary["selective_accuracy"])
        and summary["selective_accuracy"] >= config.target_accuracy
        and summary["up_signals"] >= config.min_side_signals
        and summary["down_signals"] >= config.min_side_signals
    )
    summary["certified_80"] = bool(
        summary["passed_80"]
        and np.isfinite(summary["selective_lower"])
        and summary["selective_lower"] >= config.target_accuracy
    )
    return summary


def fit_production_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    max_train_days: int | None,
    config: ResearchConfig,
) -> dict[str, Pipeline]:
    labeled_dates = pd.DatetimeIndex(sorted(pd.unique(dataset.loc[dataset["target"].notna(), "date"])))
    if max_train_days is not None and len(labeled_dates) > max_train_days:
        labeled_dates = labeled_dates[-max_train_days:]
    train = dataset[dataset["date"].isin(labeled_dates) & dataset["target"].notna()]
    models = build_models(config.mode, config.random_state)
    fitted, _ = fit_predict_models(
        models,
        train[feature_columns],
        train["target"].astype(int),
        train[feature_columns].iloc[:1],
    )
    return fitted


def latest_prediction(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    fitted_models: dict[str, Pipeline],
    stacker: Pipeline,
    selector: Any,
    selector_columns: list[str],
    threshold: float,
    config: ResearchConfig,
    allow_signal: bool,
) -> dict[str, Any]:
    latest_candidates = dataset[(dataset["ticker"] == config.ticker) & dataset["is_latest"]].sort_values("date")
    if latest_candidates.empty:
        latest_candidates = dataset[dataset["ticker"] == config.ticker].sort_values("date").tail(1)
    latest = latest_candidates.tail(1)
    raw = predict_with_models(fitted_models, latest[feature_columns])
    probability = float(apply_stacker(stacker, raw)[0])
    selector_x = build_selector_features(
        latest.reset_index(drop=True),
        raw.reset_index(drop=True),
        np.array([probability]),
        selector_columns,
    )
    selection_score = float(selector.predict_proba(selector_x)[0, 1])
    model_selected = selection_score >= threshold
    selected = model_selected and allow_signal
    direction = "상승" if probability >= 0.5 else "하락"
    return {
        "date": str(latest["date"].iloc[0].date()),
        "close": float(latest["close_price"].iloc[0]),
        "direction_probability_up": probability,
        "direction": direction,
        "selection_score": selection_score,
        "model_selected": bool(model_selected),
        "selected": bool(selected),
        "decision": direction if selected else "관망",
    }


def save_outputs(
    output_dir: Path,
    config: ResearchConfig,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    research_table: pd.DataFrame,
    best_summary: dict[str, Any],
    best_components: dict[str, Any],
    test_predictions: pd.DataFrame,
    test_summary: dict[str, Any],
    latest: dict[str, Any],
    production_models: dict[str, Pipeline],
    failures: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    research_table.to_csv(output_dir / "configuration_results.csv", index=False, encoding="utf-8-sig")
    best_components["oof"].to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    best_components["risk_coverage"].to_csv(output_dir / "calibration_risk_coverage.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(best_components["fold_records"]).to_csv(output_dir / "fold_results.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "config": asdict(config),
        "best_configuration": best_summary,
        "test": test_summary,
        "latest": latest,
        "feature_count": len(feature_columns),
        "dataset_rows": int(len(dataset)),
        "download_failures": failures,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2, allow_nan=True)
    bundle = {
        "config": asdict(config),
        "feature_columns": feature_columns,
        "max_train_days": best_summary["max_train_days"],
        "selection_threshold": best_summary["threshold"],
        "stacker": best_components["stacker"],
        "selector": best_components["selector"],
        "selector_columns": best_components["selector_columns"],
        "models": production_models,
        "test_summary": test_summary,
    }
    joblib.dump(bundle, output_dir / "model_bundle.joblib")


def print_summary(
    config: ResearchConfig,
    best_summary: dict[str, Any],
    test_summary: dict[str, Any],
    latest: dict[str, Any],
    output_dir: Path,
) -> None:
    print()
    print("=" * 72)
    print("80% 목표 주가 방향 예측 연구 결과")
    print("=" * 72)
    print(f"종목: {config.ticker}")
    print(f"선택된 학습창: {best_summary['max_train_days'] or '확장형'} 거래일")
    print(f"검증 선택 방식: {best_summary['selection_status']}")
    print(f"검증 강제 적중률: {best_summary['forced_accuracy'] * 100:.2f}%")
    print(f"검증 선택 적중률: {best_summary['accuracy'] * 100:.2f}%")
    print(f"검증 선택 비율: {best_summary['coverage'] * 100:.2f}%")
    print(f"선택 임계값: {best_summary['threshold']:.4f}")
    print("-" * 72)
    print(f"테스트 기간: {test_summary['test_start']} ~ {test_summary['test_end']}")
    print(f"테스트 강제 적중률: {test_summary['forced_accuracy'] * 100:.2f}%")
    print(f"다수 클래스 기준선: {test_summary['majority_baseline'] * 100:.2f}%")
    print(f"테스트 선택 적중률: {test_summary['selective_accuracy'] * 100:.2f}%" if np.isfinite(test_summary['selective_accuracy']) else "테스트 선택 적중률: 신호 없음")
    print(f"테스트 선택 비율: {test_summary['coverage'] * 100:.2f}%")
    print(f"테스트 선택 신호 수: {test_summary['signals']}")
    if np.isfinite(test_summary["selective_lower"]):
        print(
            f"선택 적중률 {config.confidence_level * 100:.0f}% 구간: "
            f"{test_summary['selective_lower'] * 100:.2f}% ~ {test_summary['selective_upper'] * 100:.2f}%"
        )
    print(f"80% 독립 테스트 통과: {'예' if test_summary['passed_80'] else '아니오'}")
    print(f"80% 신뢰구간 인증: {'예' if test_summary['certified_80'] else '아니오'}")
    print("-" * 72)
    print(f"최신 데이터 날짜: {latest['date']}")
    print(f"최근 종가: {latest['close']:,.2f}")
    print(f"상승 확률: {latest['direction_probability_up'] * 100:.2f}%")
    print(f"선택 점수: {latest['selection_score'] * 100:.2f}%")
    print(f"최종 판단: {latest['decision']}")
    if not test_summary["passed_80"]:
        print("상태: 80%가 미검증되어 실전 배포 금지")
    elif not test_summary["certified_80"]:
        print("상태: 점추정 80%는 통과했지만 통계 인증 전이므로 관망 고정")
    elif not latest["selected"]:
        print("상태: 모델은 검증을 통과했지만 오늘은 저신뢰 구간")
    else:
        print("상태: 80% 신뢰구간 인증을 통과한 고신뢰 신호")
    print(f"결과 폴더: {output_dir.resolve()}")
    print("=" * 72)


def is_notebook_runtime() -> bool:
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def clean_runtime_arguments(argv: list[str] | None = None) -> list[str]:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    cleaned_arguments: list[str] = []
    index = 0
    while index < len(raw_arguments):
        current = raw_arguments[index]
        if current == "-f" and index + 1 < len(raw_arguments):
            kernel_file = raw_arguments[index + 1]
            if kernel_file.endswith(".json") and "kernel-" in Path(kernel_file).name:
                index += 2
                continue
        if current.startswith("-f="):
            kernel_file = current.split("=", 1)[1]
            if kernel_file.endswith(".json") and "kernel-" in Path(kernel_file).name:
                index += 1
                continue
        if current.startswith(("--IPKernelApp.", "--HistoryManager.", "--Session.")):
            index += 1
            continue
        cleaned_arguments.append(current)
        index += 1
    return cleaned_arguments


def parse_arguments(argv: list[str] | None = None) -> ResearchConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="000660.KS")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument("--test-days", type=int, default=252)
    parser.add_argument("--oof-days", type=int, default=756)
    parser.add_argument("--fold-days", type=int, default=126)
    parser.add_argument("--min-train-days", type=int, default=756)
    parser.add_argument("--gap-days", type=int, default=1)
    parser.add_argument("--min-calibration-signals", type=int, default=24)
    parser.add_argument("--min-test-signals", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-side-signals", type=int, default=6)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument("--refit-days", type=int, default=21)
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--output-dir", default="stock80_output")
    parser.add_argument("--random-state", type=int, default=42)
    arguments = parser.parse_args(clean_runtime_arguments(argv))
    return ResearchConfig(
        ticker=arguments.ticker.strip().upper(),
        start=arguments.start,
        target_accuracy=arguments.target_accuracy,
        test_days=arguments.test_days,
        oof_days=arguments.oof_days,
        fold_days=arguments.fold_days,
        min_train_days=arguments.min_train_days,
        gap_days=arguments.gap_days,
        min_calibration_signals=arguments.min_calibration_signals,
        min_test_signals=arguments.min_test_signals,
        min_coverage=arguments.min_coverage,
        min_side_signals=arguments.min_side_signals,
        confidence_level=arguments.confidence_level,
        refit_days=arguments.refit_days,
        mode=arguments.mode,
        output_dir=arguments.output_dir,
        random_state=arguments.random_state,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_arguments(argv)
    output_dir = Path(config.output_dir)
    panel_tickers, context_specification = get_universe(config.ticker, config.mode)
    download_tickers = unique_ordered(panel_tickers + [ticker for ticker, _ in context_specification.values()])
    histories, failures = download_universe(download_tickers, config.start)
    if config.ticker not in histories:
        raise RuntimeError(f"대상 종목 {config.ticker} 데이터를 받지 못했습니다.")
    panel_histories = {ticker: histories[ticker] for ticker in panel_tickers if ticker in histories}
    context_histories = {
        prefix: (histories[ticker], lag)
        for prefix, (ticker, lag) in context_specification.items()
        if ticker in histories
    }
    if len(panel_histories) < 4:
        raise RuntimeError("패널 종목이 4개 미만입니다. 네트워크 또는 종목 코드를 확인하세요.")
    dataset, feature_columns = build_panel_dataset(config.ticker, panel_histories, context_histories)
    labeled_target = dataset[(dataset["ticker"] == config.ticker) & dataset["target"].notna()].sort_values("date")
    target_dates = pd.DatetimeIndex(pd.unique(labeled_target["date"]))
    required_dates = config.test_days + config.gap_days + config.min_train_days + config.oof_days // 2
    if len(target_dates) < required_dates:
        raise RuntimeError(
            f"대상 종목 거래일이 부족합니다. 필요 약 {required_dates}일, 현재 {len(target_dates)}일입니다."
        )
    test_dates = target_dates[-config.test_days :]
    development_end_position = len(target_dates) - config.test_days - config.gap_days
    development_dates = target_dates[:development_end_position]
    window_candidates: list[int | None]
    if config.mode == "full":
        window_candidates = [756, 1260, 1890, None]
    else:
        window_candidates = [1260, None]
    research_summaries: list[dict[str, Any]] = []
    components_by_window: dict[str, dict[str, Any]] = {}
    for max_train_days in window_candidates:
        if max_train_days is not None and max_train_days < config.min_train_days:
            continue
        try:
            summary, components = prepare_research_configuration(
                dataset,
                feature_columns,
                development_dates,
                max_train_days,
                config,
            )
            research_summaries.append(summary)
            components_by_window[str(max_train_days)] = components
            print(
                f"후보 완료: 창={max_train_days or 'expanding'}, "
                f"선택 적중률={summary['accuracy'] * 100:.2f}%, "
                f"선택 비율={summary['coverage'] * 100:.2f}%, "
                f"상태={summary['selection_status']}"
            )
        except Exception as exc:
            print(f"후보 실패: 창={max_train_days or 'expanding'} 이유={exc}")
    if not research_summaries:
        raise RuntimeError("유효한 연구 구성을 만들지 못했습니다.")
    best_summary = max(research_summaries, key=lambda item: configuration_score(item, config))
    best_components = components_by_window[str(best_summary["max_train_days"])]
    test_predictions, _ = walk_forward_test(
        dataset,
        feature_columns,
        development_dates,
        test_dates,
        best_summary["max_train_days"],
        best_components["stacker"],
        best_components["selector"],
        best_components["selector_columns"],
        best_summary["threshold"],
        config,
    )
    test_summary = summarize_test(test_predictions, config)
    production_models = fit_production_models(
        dataset,
        feature_columns,
        best_summary["max_train_days"],
        config,
    )
    latest = latest_prediction(
        dataset,
        feature_columns,
        production_models,
        best_components["stacker"],
        best_components["selector"],
        best_components["selector_columns"],
        best_summary["threshold"],
        config,
        test_summary["certified_80"],
    )
    research_table = pd.DataFrame(research_summaries)
    save_outputs(
        output_dir,
        config,
        dataset,
        feature_columns,
        research_table,
        best_summary,
        best_components,
        test_predictions,
        test_summary,
        latest,
        production_models,
        failures,
    )
    print_summary(config, best_summary, test_summary, latest, output_dir)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("중단되었습니다.", file=sys.stderr)
        if is_notebook_runtime():
            raise
        raise SystemExit(130)
    except Exception as error:
        print(f"오류: {error}", file=sys.stderr)
        if is_notebook_runtime():
            raise
        raise SystemExit(1)
    if not is_notebook_runtime():
        raise SystemExit(exit_code)
