"""
Hyperparameter Tuning Module
================================
Uses Optuna to optimize hyperparameters for LightGBM, XGBoost, CatBoost, and MLP.
Training Data: 2024
Validation Data: 2025
(Test set runs 2026 onwards in the rolling loop)

Usage:
    python src/models/tune_hyperparams.py

Output:
    /data/processed/best_params.json
"""

import sys
import json
import logging
from pathlib import Path
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "tuning.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "actual_load_mw", "wind_forecast_mw", "solar_forecast_mw",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "price_lag_24h", "price_lag_168h", "load_lag_24h",
    "renewable_penetration", "price_rolling_mean_24h",
    "price_rolling_std_24h", "net_load_mw",
]

N_TRIALS = 20

def get_data_splits():
    df = pd.read_csv(PROCESSED_DIR / "model_ready.csv", index_col="timestamp", parse_dates=True)
    df = df.dropna(subset=FEATURE_COLS + ["price_eur_mwh"])
    
    # Train: 2024
    train = df[(df.index >= "2024-01-01") & (df.index < "2025-01-01")]
    # Valid: 2025
    valid = df[(df.index >= "2025-01-01") & (df.index < "2026-01-01")]

    X_train = train[FEATURE_COLS]
    y_train = train["price_eur_mwh"].values
    X_valid = valid[FEATURE_COLS]
    y_valid = valid["price_eur_mwh"].values

    return X_train, y_train, X_valid, y_valid

def scale_data(X_train, X_valid):
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_valid)
    return X_t, X_v

# ---------------------------------------------------------------------------
# Optuna Objectives
# ---------------------------------------------------------------------------
def objective_lgbm(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def objective_xgb(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "n_estimators": 200,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def objective_cb(trial, X_train, y_train, X_valid, y_valid):
    if cb is None:
        raise ImportError("CatBoost not installed")
    params = {
        "iterations": 200,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 10, log=True),
        "random_seed": 42,
        "verbose": 0,
    }
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

def objective_mlp(trial, X_train_s, y_train, X_valid_s, y_valid):
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64)])
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": 200,
        "early_stopping": True,
        "random_state": 42,
    }
    model = MLPRegressor(**params)
    model.fit(X_train_s, y_train)
    preds = model.predict(X_valid_s)
    return mean_absolute_error(y_valid, preds)

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning via Optuna")
    logger.info("=" * 60)

    X_train, y_train, X_valid, y_valid = get_data_splits()
    logger.info(f"Train (2024): {X_train.shape[0]} rows | Valid (2025): {X_valid.shape[0]} rows")
    
    best_params = {}

    # 1. LightGBM
    logger.info("Tuning LightGBM...")
    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(lambda t: objective_lgbm(t, X_train, y_train, X_valid, y_valid), n_trials=N_TRIALS)
    best_params["lgbm"] = study_lgbm.best_params
    logger.info(f"  Best MAE: {study_lgbm.best_value:.2f} | Params: {study_lgbm.best_params}")

    # 2. XGBoost
    logger.info("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda t: objective_xgb(t, X_train, y_train, X_valid, y_valid), n_trials=N_TRIALS)
    best_params["xgb"] = study_xgb.best_params
    logger.info(f"  Best MAE: {study_xgb.best_value:.2f} | Params: {study_xgb.best_params}")

    # 3. CatBoost
    if cb is not None:
        logger.info("Tuning CatBoost...")
        study_cb = optuna.create_study(direction="minimize")
        study_cb.optimize(lambda t: objective_cb(t, X_train, y_train, X_valid, y_valid), n_trials=N_TRIALS)
        best_params["catboost"] = study_cb.best_params
        logger.info(f"  Best MAE: {study_cb.best_value:.2f} | Params: {study_cb.best_params}")

    # 4. MLP
    logger.info("Tuning MLP...")
    X_t_s, X_v_s = scale_data(X_train, X_valid)
    study_mlp = optuna.create_study(direction="minimize")
    study_mlp.optimize(lambda t: objective_mlp(t, X_t_s, y_train, X_v_s, y_valid), n_trials=N_TRIALS)
    # Tuples to lists for JSON serialization
    mlp_p = study_mlp.best_params
    mlp_p["hidden_layer_sizes"] = list(mlp_p["hidden_layer_sizes"])
    best_params["mlp"] = mlp_p
    logger.info(f"  Best MAE: {study_mlp.best_value:.2f} | Params: {study_mlp.best_params}")

    # Save
    out_path = PROCESSED_DIR / "best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Saved optimal parameters to {out_path}")

if __name__ == "__main__":
    main()
