from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "girl.csv"
OPTUNA_DB_PATH = ROOT / "data" / "q4_optuna_studies.db"
MODEL_DIR = ROOT / "data" / "q4_models"
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_DB_PATH.as_posix()}"

COL_SUBJECT = "孕妇代码"
COL_AGE = "年龄"
COL_HEIGHT = "身高"
COL_WEIGHT = "体重"
COL_GA = "检测孕周"
COL_IVF = "IVF妊娠"
COL_BMI = "孕妇BMI"
COL_READS = "原始读段数"
COL_ALIGN_RATIO = "在参考基因组上比对的比例"
COL_DUP_RATIO = "重复读段的比例"
COL_UNIQUE = "唯一比对的读段数"
COL_GC = "GC含量"
COL_Z13 = "13号染色体的Z值"
COL_Z18 = "18号染色体的Z值"
COL_Z21 = "21号染色体的Z值"
COL_ZX = "X染色体的Z值"
COL_X_CONC = "X染色体浓度"
COL_GC13 = "13号染色体的GC含量"
COL_GC18 = "18号染色体的GC含量"
COL_GC21 = "21号染色体的GC含量"
COL_FILTERED_RATIO = "被过滤掉读段数的比例"
COL_AB = "染色体的非整倍体"
COL_PREGNANCY_TIMES = "怀孕次数"
COL_PARITY_TIMES = "生产次数"

CLASS_NAMES = ["Normal", "T13", "T18", "T13T18", "T21", "T13T21", "T18T21", "T13T18T21"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

STATE_CODE_TO_NAME = {
    0: "Normal",
    1: "T13",
    2: "T18",
    3: "T13T18",
    4: "T21",
    5: "T13T21",
    6: "T18T21",
    7: "T13T18T21",
}

EXPERT_CONFIG = {
    "T13": {"target": "is_abnormal_13", "prob_col": "P_T13"},
    "T18": {"target": "is_abnormal_18", "prob_col": "P_T18"},
    "T21": {"target": "is_abnormal_21", "prob_col": "P_T21"},
    "complex": {"target": "is_multi_abnormal", "prob_col": "P_complex"},
}

CORE_META_FEATURES = [
    "age",
    "bmi",
    "ga",
    "z13",
    "z18",
    "z21",
    "zx",
    "gc",
    "gc13",
    "gc18",
    "gc21",
    "reads_log",
    "unique_log",
    "unique_ratio",
    "alignment_ratio",
    "duplicate_ratio",
    "filtered_ratio",
    "x_concentration",
    "height",
    "weight",
    "pregnancy_times",
    "parity_times",
    "ivf",
]

RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS_EXPERT = int(os.getenv("Q4_TRIALS_EXPERT", "20"))
N_TRIALS_META = int(os.getenv("Q4_TRIALS_META", "20"))
N_META_RFE_SUPP = 8
EARLY_STOPPING_ROUNDS = 50
SMOTE_META_RATIO = 0.5
LOGIC_Z_THRESHOLD = 3.0
STUDY_PREFIX = "q4_ref_v2"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_gestational_week(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    if "+" in s:
        left, right = s.split("+", 1)
        try:
            return float(left) + float(right) / 7.0
        except ValueError:
            return np.nan
    if "w" in s:
        s = s.replace("w", "")
        try:
            return float(s)
        except ValueError:
            return np.nan
    return np.nan


def parse_ivf_flag(value) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    if s in {"", "0", "false", "no", "none", "nan", "自然", "否"}:
        return 0
    return 1


def ab_to_flags(ab_value: str) -> Tuple[int, int, int]:
    s = str(ab_value).strip().upper()
    return int("T13" in s), int("T18" in s), int("T21" in s)


def build_modeling_data(df_raw: pd.DataFrame):
    df = df_raw.copy()
    t13, t18, t21 = zip(*df[COL_AB].apply(ab_to_flags))
    df["is_abnormal_13"] = np.array(t13, dtype=int)
    df["is_abnormal_18"] = np.array(t18, dtype=int)
    df["is_abnormal_21"] = np.array(t21, dtype=int)
    df["is_multi_abnormal"] = (df["is_abnormal_13"] + df["is_abnormal_18"] + df["is_abnormal_21"] >= 2).astype(int)

    ga_num = pd.to_numeric(df[COL_GA], errors="coerce")
    ga_num = ga_num.fillna(df[COL_GA].apply(parse_gestational_week))

    numeric = pd.DataFrame({
        "age": pd.to_numeric(df[COL_AGE], errors="coerce"),
        "height": pd.to_numeric(df[COL_HEIGHT], errors="coerce"),
        "weight": pd.to_numeric(df[COL_WEIGHT], errors="coerce"),
        "ga": ga_num,
        "bmi": pd.to_numeric(df[COL_BMI], errors="coerce"),
        "reads": pd.to_numeric(df[COL_READS], errors="coerce"),
        "alignment_ratio": pd.to_numeric(df[COL_ALIGN_RATIO], errors="coerce"),
        "duplicate_ratio": pd.to_numeric(df[COL_DUP_RATIO], errors="coerce"),
        "unique_reads": pd.to_numeric(df[COL_UNIQUE], errors="coerce"),
        "gc": pd.to_numeric(df[COL_GC], errors="coerce"),
        "z13": pd.to_numeric(df[COL_Z13], errors="coerce"),
        "z18": pd.to_numeric(df[COL_Z18], errors="coerce"),
        "z21": pd.to_numeric(df[COL_Z21], errors="coerce"),
        "zx": pd.to_numeric(df[COL_ZX], errors="coerce"),
        "x_concentration": pd.to_numeric(df[COL_X_CONC], errors="coerce"),
        "gc13": pd.to_numeric(df[COL_GC13], errors="coerce"),
        "gc18": pd.to_numeric(df[COL_GC18], errors="coerce"),
        "gc21": pd.to_numeric(df[COL_GC21], errors="coerce"),
        "filtered_ratio": pd.to_numeric(df[COL_FILTERED_RATIO], errors="coerce"),
        "pregnancy_times": pd.to_numeric(df[COL_PREGNANCY_TIMES], errors="coerce"),
        "parity_times": pd.to_numeric(df[COL_PARITY_TIMES], errors="coerce"),
    })

    x = pd.DataFrame({
        "age": numeric["age"],
        "height": numeric["height"],
        "weight": numeric["weight"],
        "ga": numeric["ga"],
        "bmi": numeric["bmi"],
        "z13": numeric["z13"],
        "z18": numeric["z18"],
        "z21": numeric["z21"],
        "zx": numeric["zx"],
        "gc": numeric["gc"],
        "gc13": numeric["gc13"],
        "gc18": numeric["gc18"],
        "gc21": numeric["gc21"],
        "reads_log": np.log1p(numeric["reads"]),
        "unique_log": np.log1p(numeric["unique_reads"]),
        "unique_ratio": numeric["unique_reads"] / (numeric["reads"] + 1e-9),
        "alignment_ratio": numeric["alignment_ratio"],
        "duplicate_ratio": numeric["duplicate_ratio"],
        "filtered_ratio": numeric["filtered_ratio"],
        "x_concentration": numeric["x_concentration"],
        "pregnancy_times": numeric["pregnancy_times"],
        "parity_times": numeric["parity_times"],
        "ivf": df[COL_IVF].apply(parse_ivf_flag).astype(float),
        "z13_z18": numeric["z13"] * numeric["z18"],
        "z13_z21": numeric["z13"] * numeric["z21"],
        "z18_z21": numeric["z18"] * numeric["z21"],
        "z13_minus_zx": numeric["z13"] - numeric["zx"],
        "z18_minus_zx": numeric["z18"] - numeric["zx"],
        "z21_minus_zx": numeric["z21"] - numeric["zx"],
        "gc13_ratio": numeric["gc13"] / (numeric["gc"] + 1e-9),
        "gc18_ratio": numeric["gc18"] / (numeric["gc"] + 1e-9),
        "gc21_ratio": numeric["gc21"] / (numeric["gc"] + 1e-9),
        "gc13_minus_gc": numeric["gc13"] - numeric["gc"],
        "gc18_minus_gc": numeric["gc18"] - numeric["gc"],
        "gc21_minus_gc": numeric["gc21"] - numeric["gc"],
    })

    target_code = (
        df["is_abnormal_21"] * 4
        + df["is_abnormal_18"] * 2
        + df["is_abnormal_13"] * 1
    ).astype(int)
    target_name = target_code.map(STATE_CODE_TO_NAME)
    y_idx = target_name.map(CLASS_TO_INDEX).to_numpy(dtype=int)

    key_cols = ["age", "bmi", "ga", "z13", "z18", "z21", "zx", "gc", "reads_log", "unique_log"]
    keep_mask = ~x[key_cols].isna().any(axis=1)
    x = x.loc[keep_mask].reset_index(drop=True)
    y_idx = y_idx[keep_mask.to_numpy()]
    df = df.loc[keep_mask].reset_index(drop=True)
    target_df = df[["is_abnormal_13", "is_abnormal_18", "is_abnormal_21", "is_multi_abnormal"]].astype(int)

    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    return x, y_idx, target_df


def get_cv_splitter(y: np.ndarray, n_splits: int, seed: int):
    counts = pd.Series(y).value_counts()
    min_count = int(counts.min())
    if min_count >= 2:
        folds = max(2, min(n_splits, min_count))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), True
    return KFold(n_splits=2, shuffle=True, random_state=seed), False


def create_or_load_study(study_name: str, seed: int):
    return optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )


def expand_proba_to_all_classes(proba: np.ndarray, classes_: np.ndarray):
    full = np.zeros((proba.shape[0], len(CLASS_NAMES)), dtype=float)
    full[:, classes_.astype(int)] = proba
    return full


def smote_resample_multiclass(x: pd.DataFrame, y: np.ndarray, seed: int):
    class_counts = pd.Series(y).value_counts()
    if class_counts.shape[0] <= 1:
        return x.copy(), y.copy()
    majority_count = int(class_counts.max())
    sampling_strategy = {}
    for cls, cnt in class_counts.items():
        if int(cnt) > 1 and int(cnt) < majority_count:
            sampling_strategy[int(cls)] = max(int(cnt), int(majority_count * SMOTE_META_RATIO))
    if not sampling_strategy:
        return x.copy(), y.copy()
    try:
        smote = SMOTE(random_state=seed, k_neighbors=1, sampling_strategy=sampling_strategy)
        x_res, y_res = smote.fit_resample(x, y)
        return pd.DataFrame(x_res, columns=x.columns), y_res
    except ValueError:
        return x.copy(), y.copy()


def tune_single_expert(x: pd.DataFrame, y: np.ndarray, study_name: str, seed: int):
    def objective(trial: optuna.trial.Trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_jobs": -1,
            "is_unbalance": True,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 10, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        cv, is_stratified = get_cv_splitter(y, N_SPLITS, seed)
        losses = []
        split_iter = cv.split(x, y) if is_stratified else cv.split(x)
        for step, (idx_tr, idx_va) in enumerate(split_iter, start=1):
            model = lgb.LGBMClassifier(**params, random_state=seed + step)
            model.fit(
                x.iloc[idx_tr],
                y[idx_tr],
                eval_set=[(x.iloc[idx_va], y[idx_va])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            pred = model.predict_proba(x.iloc[idx_va])[:, 1]
            losses.append(log_loss(y[idx_va], pred, labels=[0, 1]))
            trial.report(float(np.mean(losses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses)) if losses else 999.0

    study = create_or_load_study(study_name=study_name, seed=seed)
    complete_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS_EXPERT - complete_trials)
    if remaining > 0:
        print(f"[Optuna] {study_name}: run {remaining} trials (complete={complete_trials})")
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_jobs": -1,
        "is_unbalance": True,
    })
    final_model = lgb.LGBMClassifier(**best_params, random_state=seed)
    final_model.fit(x, y)
    return final_model, best_params


def train_expert_models(x: pd.DataFrame, target_df: pd.DataFrame, seed: int, study_tag: str):
    experts = {}
    best_param_map = {}
    for k, cfg in EXPERT_CONFIG.items():
        y = target_df[cfg["target"]].to_numpy(dtype=int)
        study_name = f"{STUDY_PREFIX}_{study_tag}_expert_{k}"
        model, best_params = tune_single_expert(x, y, study_name=study_name, seed=seed + 13 * (len(best_param_map) + 1))
        experts[k] = {"model": model, "prob_col": cfg["prob_col"]}
        best_param_map[k] = best_params
        print(f"[Expert] trained: {k}")
    return experts, best_param_map


def build_meta_probs_from_experts(x: pd.DataFrame, experts: Dict[str, dict]):
    meta = pd.DataFrame(index=x.index)
    for k, cfg in EXPERT_CONFIG.items():
        model = experts[k]["model"]
        meta[cfg["prob_col"]] = model.predict_proba(x)[:, 1]
    return meta


def build_oof_meta_features(x: pd.DataFrame, target_df: pd.DataFrame, expert_param_map: Dict[str, dict], seed: int):
    meta = pd.DataFrame(index=x.index)
    for i, (k, cfg) in enumerate(EXPERT_CONFIG.items(), start=1):
        y = target_df[cfg["target"]].to_numpy(dtype=int)
        cv, is_stratified = get_cv_splitter(y, N_SPLITS, seed + i)
        split_iter = cv.split(x, y) if is_stratified else cv.split(x)
        oof_pred = np.zeros(len(x), dtype=float)
        for fold, (idx_tr, idx_va) in enumerate(split_iter, start=1):
            model = lgb.LGBMClassifier(**expert_param_map[k], random_state=seed + i * 100 + fold)
            model.fit(x.iloc[idx_tr], y[idx_tr])
            oof_pred[idx_va] = model.predict_proba(x.iloc[idx_va])[:, 1]
        meta[cfg["prob_col"]] = oof_pred
    return meta


def select_meta_original_features(x: pd.DataFrame, y_idx: np.ndarray, seed: int):
    existing_core = [c for c in CORE_META_FEATURES if c in x.columns]
    remaining = [c for c in x.columns if c not in existing_core]
    selected_supp = []
    if remaining:
        n_select = min(N_META_RFE_SUPP, len(remaining))
        estimator = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_NAMES),
            n_estimators=220,
            num_leaves=31,
            random_state=seed,
            verbosity=-1,
        )
        rfe = RFE(estimator=estimator, n_features_to_select=n_select, step=1)
        rfe.fit(x[remaining], y_idx)
        selected_supp = list(pd.Index(remaining)[rfe.support_])
    selected = existing_core + selected_supp
    return selected, existing_core, selected_supp


def tune_meta_model(x_meta: pd.DataFrame, y_idx: np.ndarray, study_name: str, seed: int):
    def objective(trial: optuna.trial.Trial):
        params = {
            "objective": "multiclass",
            "num_class": len(CLASS_NAMES),
            "metric": "multi_logloss",
            "verbosity": -1,
            "n_jobs": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        cv, is_stratified = get_cv_splitter(y_idx, N_SPLITS, seed)
        losses = []
        split_iter = cv.split(x_meta, y_idx) if is_stratified else cv.split(x_meta)
        for step, (idx_tr, idx_va) in enumerate(split_iter, start=1):
            x_tr = x_meta.iloc[idx_tr].reset_index(drop=True)
            y_tr = y_idx[idx_tr]
            x_tr_smote, y_tr_smote = smote_resample_multiclass(x_tr, y_tr, seed + step)
            if np.unique(y_tr_smote).shape[0] < 2:
                return 999.0
            model = lgb.LGBMClassifier(**params, random_state=seed + step)
            y_va = y_idx[idx_va]
            train_labels = np.unique(y_tr_smote)
            val_labels = np.unique(y_va)
            if np.setdiff1d(val_labels, train_labels).size == 0:
                model.fit(
                    x_tr_smote,
                    y_tr_smote,
                    eval_set=[(x_meta.iloc[idx_va], y_va)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
                )
            else:
                model.fit(x_tr_smote, y_tr_smote)
            pred = model.predict_proba(x_meta.iloc[idx_va])
            pred_full = expand_proba_to_all_classes(pred, model.classes_)
            losses.append(log_loss(y_va, pred_full, labels=list(range(len(CLASS_NAMES)))))
            trial.report(float(np.mean(losses)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses)) if losses else 999.0

    study = create_or_load_study(study_name=study_name, seed=seed)
    complete_trials = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, N_TRIALS_META - complete_trials)
    if remaining > 0:
        print(f"[Optuna] {study_name}: run {remaining} trials (complete={complete_trials})")
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update({
        "objective": "multiclass",
        "num_class": len(CLASS_NAMES),
        "metric": "multi_logloss",
        "verbosity": -1,
        "n_jobs": -1,
    })
    x_final, y_final = smote_resample_multiclass(x_meta, y_idx, seed + 999)
    final_model = lgb.LGBMClassifier(**best_params, random_state=seed)
    final_model.fit(x_final, y_final)
    return final_model, best_params


def apply_logic_correction(y_pred: np.ndarray, x_base: pd.DataFrame):
    corrected = y_pred.astype(int).copy()
    z13 = x_base["z13"].to_numpy()
    z18 = x_base["z18"].to_numpy()
    z21 = x_base["z21"].to_numpy()

    idx_t13t18t21 = CLASS_TO_INDEX["T13T18T21"]
    idx_t13t18 = CLASS_TO_INDEX["T13T18"]
    idx_t13t21 = CLASS_TO_INDEX["T13T21"]
    idx_t18t21 = CLASS_TO_INDEX["T18T21"]

    mask_3 = (z13 >= LOGIC_Z_THRESHOLD) & (z18 >= LOGIC_Z_THRESHOLD) & (z21 >= LOGIC_Z_THRESHOLD)
    mask_13_18 = (z13 >= LOGIC_Z_THRESHOLD) & (z18 >= LOGIC_Z_THRESHOLD) & (~mask_3)
    mask_13_21 = (z13 >= LOGIC_Z_THRESHOLD) & (z21 >= LOGIC_Z_THRESHOLD) & (~mask_3)
    mask_18_21 = (z18 >= LOGIC_Z_THRESHOLD) & (z21 >= LOGIC_Z_THRESHOLD) & (~mask_3)

    corrected[mask_3] = idx_t13t18t21
    corrected[mask_13_18] = idx_t13t18
    corrected[mask_13_21] = idx_t13t21
    corrected[mask_18_21] = idx_t18t21
    return corrected


def build_meta_input_for_prediction(x: pd.DataFrame, experts: Dict[str, dict], selected_original_cols: list[str]):
    meta_probs = build_meta_probs_from_experts(x, experts)
    return pd.concat([meta_probs, x[selected_original_cols]], axis=1)


def train_stack_pipeline(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    target_train: pd.DataFrame,
    seed: int,
    study_tag: str,
):
    experts, expert_param_map = train_expert_models(x_train, target_train, seed=seed, study_tag=study_tag)
    meta_oof = build_oof_meta_features(x_train, target_train, expert_param_map, seed=seed + 200)
    selected_original_cols, core_cols, supp_cols = select_meta_original_features(x_train, y_train, seed=seed + 300)
    x_meta_train = pd.concat([meta_oof, x_train[selected_original_cols]], axis=1)
    meta_model, _ = tune_meta_model(
        x_meta_train,
        y_train,
        study_name=f"{STUDY_PREFIX}_{study_tag}_meta",
        seed=seed + 400,
    )
    return experts, meta_model, selected_original_cols, core_cols, supp_cols


def reference_five_fold_evaluation(
    x: pd.DataFrame,
    y_true: np.ndarray,
    experts: Dict[str, dict],
    meta_model: lgb.LGBMClassifier,
    selected_original_cols: list[str],
    seed: int,
):
    rng = np.random.RandomState(seed)
    n_splits = 5
    fold_buckets = [[] for _ in range(n_splits)]
    for cls in np.unique(y_true):
        cls_idx = np.where(y_true == cls)[0]
        rng.shuffle(cls_idx)
        for pos, sample_idx in enumerate(cls_idx):
            fold_buckets[pos % n_splits].append(int(sample_idx))

    all_idx = np.arange(len(y_true))
    folds = []
    for bucket in fold_buckets:
        idx_va = np.array(sorted(bucket), dtype=int)
        idx_tr = np.setdiff1d(all_idx, idx_va, assume_unique=False)
        folds.append((idx_tr, idx_va))

    all_true, all_pred = [], []
    fold_acc, fold_macro_f1 = [], []

    for fold_id, (_, idx_va) in enumerate(folds, start=1):
        print(f"\n[Fold {fold_id}/{len(folds)}] evaluating...")
        x_val = x.iloc[idx_va]
        y_val = y_true[idx_va]

        x_meta_val = build_meta_input_for_prediction(x_val, experts, selected_original_cols)
        x_meta_val = x_meta_val.reindex(columns=meta_model.feature_name_, fill_value=0.0)
        y_pred_primary = meta_model.predict(x_meta_val).astype(int)
        y_pred_logic = apply_logic_correction(y_pred_primary, x_val)

        all_true.append(y_val)
        all_pred.append(y_pred_logic)
        fold_acc.append(accuracy_score(y_val, y_pred_logic))
        _, _, fold_f1, _ = precision_recall_fscore_support(
            y_val,
            y_pred_logic,
            labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
        fold_macro_f1.append(float(np.mean(fold_f1)))

    return np.concatenate(all_true), np.concatenate(all_pred), fold_acc, fold_macro_f1, len(folds)


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, normal_label: int):
    y_true_bin = (y_true != normal_label).astype(int)
    y_pred_bin = (y_pred != normal_label).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary", zero_division=0)
    acc = accuracy_score(y_true_bin, y_pred_bin)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray):
    labels = list(range(len(CLASS_NAMES)))
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup,
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }


def print_binary_report(binary_metrics: dict, title: str):
    print(f"\n{title} Binary Evaluation (Normal vs Abnormal)")
    print(f"Accuracy : {binary_metrics['accuracy']:.4f}")
    print(f"Precision: {binary_metrics['precision']:.4f}")
    print(f"Recall   : {binary_metrics['recall']:.4f}")
    print(f"F1       : {binary_metrics['f1']:.4f}")


def print_multiclass_table(metrics: dict, total_samples: int, title: str):
    print(f"\n{title} Eight-Class Evaluation Summary")
    print(f"{'Class':<14}{'Precision':<12}{'Recall':<12}{'F1-score':<12}{'Support':<10}")
    print("-" * 62)
    for i, cls_name in enumerate(CLASS_NAMES):
        print(
            f"{cls_name:<14}{metrics['precision'][i]:<12.2f}{metrics['recall'][i]:<12.2f}"
            f"{metrics['f1'][i]:<12.2f}{int(metrics['support'][i]):<10}"
        )
    print("-" * 62)
    print(f"Accuracy     {metrics['accuracy']:.2f}")
    print(f"Total samples {total_samples}")


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(f"\n{title} Confusion Matrix")
    print(cm_df.to_string())


def report_results(title: str, y_true: np.ndarray, y_pred: np.ndarray, normal_label: int):
    binary_metrics = evaluate_binary(y_true, y_pred, normal_label=normal_label)
    multiclass_metrics = evaluate_multiclass(y_true, y_pred)
    print_binary_report(binary_metrics, title=title)
    print(f"\n{title} Macro Metrics")
    print(f"Macro Precision: {multiclass_metrics['macro_precision']:.4f}")
    print(f"Macro Recall   : {multiclass_metrics['macro_recall']:.4f}")
    print(f"Macro F1       : {multiclass_metrics['macro_f1']:.4f}")
    print_multiclass_table(multiclass_metrics, total_samples=len(y_true), title=title)
    print_confusion(y_true, y_pred, title=title)


def persist_models(
    experts: Dict[str, dict],
    meta_model: lgb.LGBMClassifier,
    selected_original_cols: list[str],
    scaler: StandardScaler,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for k, pack in experts.items():
        save_path = MODEL_DIR / f"{k}_expert.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(pack["model"], f)
    with open(MODEL_DIR / "meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    with open(MODEL_DIR / "meta_original_cols.pkl", "wb") as f:
        pickle.dump(selected_original_cols, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def main():
    print("[Step 1] build modeling data")
    df_raw = pd.read_csv(DATA_PATH, encoding="utf-8")
    x_raw, y_idx, target_df = build_modeling_data(df_raw)
    normal_label = CLASS_TO_INDEX["Normal"]

    print("[Step 2] full-data training")
    scaler_full = StandardScaler()
    x_full = pd.DataFrame(scaler_full.fit_transform(x_raw), columns=x_raw.columns, index=x_raw.index)
    experts_full, meta_model_full, selected_original_cols, core_cols, supp_cols = train_stack_pipeline(
        x_full,
        y_idx,
        target_df,
        seed=RANDOM_STATE + 9000,
        study_tag="full",
    )
    print(f"Meta core features: {len(core_cols)}")
    print(f"Meta supplementary RFE features: {len(supp_cols)}")
    persist_models(experts_full, meta_model_full, selected_original_cols, scaler_full)

    print("\n[Step 3] 5-fold CV evaluation (reference style)")
    y_true_cv, y_pred_cv, fold_acc, fold_macro_f1, actual_folds = reference_five_fold_evaluation(
        x_full,
        y_idx,
        experts_full,
        meta_model_full,
        selected_original_cols,
        seed=RANDOM_STATE + 500,
    )
    print(f"{actual_folds}-fold CV Overview")
    print(f"Fold Accuracy mean+/-std: {np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")
    print(f"Fold Macro-F1 mean+/-std: {np.mean(fold_macro_f1):.4f} +/- {np.std(fold_macro_f1):.4f}")
    report_results("Cross-Validation", y_true_cv, y_pred_cv, normal_label)

    print("\n[Step 4] full-data fit report")
    x_meta_full = build_meta_input_for_prediction(x_full, experts_full, selected_original_cols)
    x_meta_full = x_meta_full.reindex(columns=meta_model_full.feature_name_, fill_value=0.0)
    y_pred_full_primary = meta_model_full.predict(x_meta_full).astype(int)
    y_pred_full = apply_logic_correction(y_pred_full_primary, x_full)
    report_results("Full-Data Fit", y_idx, y_pred_full, normal_label)


if __name__ == "__main__":
    main()
