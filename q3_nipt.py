"""
Q3: NIPT timing optimization with age/gravida/para covariates.

summary:
1) Extend q2 timing optimization with covariate offsets and age-layer baseline shifts.
2) Solve a 5x5 age-layer x BMI-segment schedule with layer-wise DP and monotonic constraints.
3) Report weighted recommendations and sensitivity to kappa and q.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

import q1_lmm
from q2_nipt import (
    BMI_STEP,
    COL_BMI,
    COL_GA,
    K_SEG as K_BMI,
    compute_T_bayes_conservative,
    compute_T_star_curve,
    get_t_grid,
    make_p_kappa_and_r_fail,
)

COL_SUBJECT = q1_lmm.COL_SUBJECT
COL_FF = q1_lmm.COL_FF

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boy.csv"

COL_AGE = "年龄"
COL_GRAVIDA = "怀孕次数"
COL_PARA = "生产次数"

K_AGE = 5
Q3_T_MIN = 12.0
Q3_T_MAX = 28.0
Q3_ALPHA = 0.02
Q3_BETA = 0.12

GAMMA_BOUND = 0.5
RHO_L2 = 1e-3
RHO_L1 = 1e-4
GAMMA_ANCHOR = 2.0
DELTA_MAX = 8.0
DELTA_MIN_INC = 1.0
DELTA_Q = 75.0
SEG_MIN_INC = 0.3
AGE_MONO_MIN_INC = 0.4


def q3_phi_time_risk(t):
    """Q3 time-risk: 0 for <=12, alpha*(t-12) for (12,28), 16*alpha+beta*(t-28) for >=28."""
    t = np.asarray(t, dtype=float)
    return np.where(
        t <= 12.0,
        0.0,
        np.where(t < 28.0, Q3_ALPHA * (t - 12.0), 16.0 * Q3_ALPHA + Q3_BETA * (t - 28.0)),
    )


def enforce_monotonicity_q3(t_list):
    """Enforce non-decreasing segment times and clip to [Q3_T_MIN, Q3_T_MAX]."""
    t_arr = np.array(t_list, dtype=float)
    t_arr = np.clip(t_arr, Q3_T_MIN, Q3_T_MAX)
    for i in range(1, len(t_arr)):
        t_arr[i] = max(t_arr[i], t_arr[i - 1] + SEG_MIN_INC)
    t_arr = np.clip(t_arr, Q3_T_MIN, Q3_T_MAX)
    return t_arr.tolist()


def load_subjects_with_covariates():
    """Aggregate subject-level covariates and center them."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    use = [COL_SUBJECT, COL_AGE, COL_BMI, COL_GA, COL_FF, COL_GRAVIDA, COL_PARA]
    df = df[use].copy()
    df = df.dropna(subset=[COL_GA, COL_BMI, COL_FF])
    df = df.astype({COL_GA: float, COL_BMI: float, COL_FF: float})
    df[COL_AGE] = pd.to_numeric(df[COL_AGE], errors="coerce")
    df[COL_GRAVIDA] = pd.to_numeric(df[COL_GRAVIDA], errors="coerce")
    df[COL_PARA] = pd.to_numeric(df[COL_PARA], errors="coerce")

    # Collapse repeated test records to one subject profile for q3 covariate-aware scheduling.
    agg = (
        df.groupby(COL_SUBJECT)
        .agg(
            age=(COL_AGE, "first"),
            bmi=(COL_BMI, "mean"),
            gravida=(COL_GRAVIDA, "first"),
            para=(COL_PARA, "first"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["age", "bmi"])

    age_bar = float(agg["age"].mean())
    gravida_bar = float(agg["gravida"].mean()) if not agg["gravida"].isna().all() else 0.0
    para_bar = float(agg["para"].mean()) if not agg["para"].isna().all() else 0.0
    # Center covariates so gamma offsets are interpretable as local adjustments around the cohort average.
    agg["age_c"] = agg["age"] - age_bar
    agg["gravida_c"] = (agg["gravida"] - gravida_bar).fillna(0.0)
    agg["para_c"] = (agg["para"] - para_bar).fillna(0.0)
    return agg


def fixed_age_layer_specs():
    """Fixed age layers: <25, 25-29, 30-34, 35-39, >=40."""
    labels = ["<25", "25-29", "30-34", "35-39", ">=40"]

    def mask_fn(ages, layer):
        if layer == 0:
            return ages < 25
        if layer == 1:
            return (ages >= 25) & (ages < 30)
        if layer == 2:
            return (ages >= 30) & (ages < 35)
        if layer == 3:
            return (ages >= 35) & (ages < 40)
        return ages >= 40

    return labels, mask_fn


def estimate_gamma_offset(subjects, b_grid, t_grid, p_kappa, r_fail, q):
    """
    Estimate gamma with two stages:
    1) constrained ridge fit to subject-level timing shifts;
    2) risk refinement anchored to stage-1 result.
    """
    # Start from the q2-style monotone threshold curve as the covariate-free baseline schedule.
    t_curve, _, _ = compute_T_star_curve(b_grid, t_grid, p_kappa, q)

    b = subjects["bmi"].to_numpy(dtype=float)
    z = subjects[["age_c", "gravida_c", "para_c"]].to_numpy(dtype=float)
    t_base = np.interp(b, b_grid, t_curve)

    # Estimate each subject's earliest feasible timing under the shared p_kappa model.
    p_all = p_kappa(t_grid[:, None], b[None, :])
    feasible = p_all >= q
    any_feasible = feasible.any(axis=0)
    idx_star = np.where(any_feasible, feasible.argmax(axis=0), len(t_grid) - 1)
    t_star = t_grid[idx_star]
    shift_target = np.clip(t_star - t_base, -GAMMA_BOUND, GAMMA_BOUND)

    bounds = [
        (-GAMMA_BOUND, 0.0),         # age: older -> later
        (0.0, GAMMA_BOUND),          # gravida: more pregnancies -> earlier
        (-GAMMA_BOUND, 0.0),         # para: more births -> later
    ]

    # Stage 1 fits a stable bounded linear offset before risk-based refinement.
    def ls_obj(gamma):
        pred = z @ gamma
        return float(np.mean((pred - shift_target) ** 2) + RHO_L2 * np.sum(gamma ** 2))

    ls_res = minimize(ls_obj, x0=np.zeros(3), method="L-BFGS-B", bounds=bounds)
    gamma_ls = ls_res.x if ls_res.success else np.zeros(3)

    # Stage 2 optimizes q3 risk while staying close to the stage-1 estimate.
    def risk_obj(gamma):
        t_adj = np.clip(t_base + (z @ gamma), Q3_T_MIN, Q3_T_MAX)
        risk = np.mean(r_fail(t_adj, b) + q3_phi_time_risk(t_adj))
        anchor = GAMMA_ANCHOR * np.sum((gamma - gamma_ls) ** 2)
        l1 = RHO_L1 * np.sum(np.abs(gamma))
        return float(risk + anchor + l1)

    opt_res = minimize(risk_obj, x0=gamma_ls, method="L-BFGS-B", bounds=bounds)
    return opt_res.x if opt_res.success else gamma_ls


def estimate_age_layer_offsets(subjects, p_kappa, q, t_grid):
    """
    Estimate layer-wise baseline offsets delta_age[layer] and enforce monotone increase by age.
    delta_age[0] is anchored at 0.
    When chained q1->q2->q3 signal is weak, use a mild minimum-increment prior to avoid degenerate equal layers.
    """
    _, age_mask_fn = fixed_age_layer_specs()
    ages = subjects["age"].to_numpy(dtype=float)
    layer_raw = np.zeros(K_AGE, dtype=float)

    # Estimate a layer baseline from high-percentile feasible times to reflect conservative scheduling.
    for layer in range(K_AGE):
        mask = age_mask_fn(ages, layer)
        s = subjects.loc[mask]
        if s.empty:
            layer_raw[layer] = layer_raw[layer - 1] if layer > 0 else 0.0
            continue
        b = s["bmi"].to_numpy(dtype=float)
        p_all = p_kappa(t_grid[:, None], b[None, :])
        feasible = p_all >= q
        any_feasible = feasible.any(axis=0)
        idx_star = np.where(any_feasible, feasible.argmax(axis=0), len(t_grid) - 1)
        t_star = t_grid[idx_star]
        layer_raw[layer] = float(np.percentile(t_star, DELTA_Q))

    # Convert to positive delay magnitude, then map back to negative shift.
    raw_delta = layer_raw - layer_raw[0]
    raw_delta = np.clip(raw_delta, 0.0, DELTA_MAX)
    # Smooth layer offsets with isotonic regression so age effects remain ordered without jagged jumps.
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    x = np.arange(K_AGE, dtype=float)
    delta = iso.fit_transform(x, raw_delta)
    delta = np.clip(delta - delta[0], 0.0, DELTA_MAX)
    # Weak prior to preserve age stratification when empirical deltas collapse.
    for i in range(1, K_AGE):
        delta[i] = max(delta[i], delta[i - 1] + DELTA_MIN_INC)
    if delta[-1] > DELTA_MAX:
        scale = DELTA_MAX / max(delta[-1], 1e-9)
        delta = delta * scale
        for i in range(1, K_AGE):
            delta[i] = max(delta[i], delta[i - 1] + min(DELTA_MIN_INC * scale, DELTA_MIN_INC))
        delta = np.clip(delta, 0.0, DELTA_MAX)
    return -np.asarray(delta, dtype=float)


def _precompute_layer_costs(b_grid, b_subj, shift_subj, t_grid, p_kappa, q):
    """Precompute best segment cost/time for all BMI-grid ranges [i:j) using subject-level shifts."""
    n = len(b_grid)
    best_cost = np.full((n + 1, n + 1), np.inf)
    best_t = np.full((n + 1, n + 1), np.nan)
    # Use a minimum width in BMI-grid units to avoid unstable micro-segments.
    min_seg = max(3, int(round(1.0 / BMI_STEP)))
    min_seg = min(min_seg, max(1, n // K_BMI))
    n_total = max(1, len(b_subj))

    # Conservative floor uses T_final = max(T_bayes, T*).
    t_curve, t_star_raw, _ = compute_T_star_curve(b_grid, t_grid, p_kappa, q)
    t_floor_curve = compute_T_bayes_conservative(b_grid, t_star_raw, t_curve)

    for i in range(n):
        for j in range(i + min_seg, n + 1):
            # Map the grid segment to real subjects in this age layer for subject-level cost evaluation.
            lo, hi = float(b_grid[i]), float(b_grid[j - 1])
            mask_seg = (b_subj >= lo) & (b_subj <= hi)
            if not np.any(mask_seg):
                continue

            b_seg = b_subj[mask_seg]
            sh_seg = shift_subj[mask_seg]
            t_adj = np.clip(t_grid[:, None] + sh_seg[None, :], Q3_T_MIN, Q3_T_MAX)
            p_seg = p_kappa(t_adj, b_seg[None, :])
            min_p = np.min(p_seg, axis=1)
            risk_vec = (1.0 - p_seg) + q3_phi_time_risk(t_adj)
            # Mass weighting keeps segments with more subjects more influential in the layer objective.
            mass = b_seg.size / n_total
            cost_vec = mass * np.mean(risk_vec, axis=1)
            feasible = np.flatnonzero(min_p >= q)
            if feasible.size > 0:
                k = feasible[int(np.argmin(cost_vec[feasible]))]
                best_cost[i, j] = float(cost_vec[k])
                best_t[i, j] = float(t_grid[k])
            else:
                k = int(np.argmax(min_p))
                best_cost[i, j] = float(cost_vec[k] + 1e3 * mass)
                best_t[i, j] = float(t_grid[k])
    return best_cost, best_t, min_seg, t_floor_curve


def _best_time_for_segment(t_grid, b_seg, shift_seg, p_kappa, q):
    """Select best feasible timing for one segment under subject-level gamma offsets."""
    t_adj = np.clip(t_grid[:, None] + shift_seg[None, :], Q3_T_MIN, Q3_T_MAX)
    p_mat = p_kappa(t_adj, b_seg[None, :])
    min_p = np.min(p_mat, axis=1)
    cost = np.mean((1.0 - p_mat) + q3_phi_time_risk(t_adj), axis=1)
    feasible = np.flatnonzero(min_p >= q)
    if feasible.size > 0:
        local = feasible[int(np.argmin(cost[feasible]))]
        return float(t_grid[local])
    local = int(np.argmax(min_p))
    return float(t_grid[local])


def _dp_layer_fixed_k(b_grid, best_cost, best_t, min_seg, k_seg=K_BMI):
    """DP solve for fixed K contiguous BMI segments with non-decreasing timing."""
    n = len(b_grid)
    dp = np.full((n + 1, k_seg + 1), np.inf)
    parent = np.full((n + 1, k_seg + 1), -1, dtype=int)
    last_t = np.full((n + 1, k_seg + 1), np.nan)
    dp[0, 0] = 0.0

    # Initialize the one-segment case, then extend to K segments with monotone-time transitions.
    for j in range(min_seg, n + 1):
        dp[j, 1] = best_cost[0, j]
        parent[j, 1] = 0
        last_t[j, 1] = best_t[0, j]

    for s in range(2, k_seg + 1):
        for j in range(s * min_seg, n + 1):
            best_val = np.inf
            best_i = -1
            best_tij = np.nan
            for i in range((s - 1) * min_seg, j - min_seg + 1):
                if not np.isfinite(dp[i, s - 1]) or not np.isfinite(best_cost[i, j]):
                    continue
                t_prev = last_t[i, s - 1]
                t_curr = best_t[i, j]
                if np.isfinite(t_prev) and t_curr < t_prev:
                    continue
                cand = dp[i, s - 1] + best_cost[i, j]
                if cand < best_val:
                    best_val = cand
                    best_i = i
                    best_tij = t_curr
            if best_i >= 0:
                dp[j, s] = best_val
                parent[j, s] = best_i
                last_t[j, s] = best_tij

    if not np.isfinite(dp[n, k_seg]):
        return None

    # Backtrack the fixed-K partition after the DP table is filled.
    breaks = [n]
    j = n
    for s in range(k_seg, 0, -1):
        i = parent[j, s]
        breaks.append(i)
        j = i
    breaks.reverse()

    segment_times = []
    segment_intervals = []
    for idx in range(k_seg):
        i_start, i_end = int(breaks[idx]), int(breaks[idx + 1])
        segment_times.append(float(best_t[i_start, i_end]))
        segment_intervals.append((float(b_grid[i_start]), float(b_grid[i_end - 1])))
    segment_times = enforce_monotonicity_q3(segment_times)
    return breaks, segment_times, segment_intervals


def _solve_one_age_layer(subjects_layer, p_kappa, q, gamma, delta_layer, t_grid):
    """Solve one age layer: its own BMI-grid 5-segment partition + recommended times."""
    layer = subjects_layer.sort_values("bmi").reset_index(drop=True)
    if layer.empty:
        raise ValueError("Empty age layer encountered under fixed age bins.")
    b_subj = layer["bmi"].to_numpy(dtype=float)
    # Subject-level timing shift combines centered covariates with the layer baseline offset.
    shift_subj = (layer[["age_c", "gravida_c", "para_c"]].to_numpy(dtype=float) @ gamma) + float(delta_layer)
    b_grid = np.arange(float(b_subj.min()), float(b_subj.max()) + 1e-9, BMI_STEP)
    # Force at least K grid points so the fixed-5-segment DP remains well-defined.
    if len(b_grid) < K_BMI:
        b_grid = np.linspace(float(b_subj.min()), float(b_subj.max()), K_BMI)

    best_cost, best_t, min_seg, t_floor_curve = _precompute_layer_costs(
        b_grid, b_subj, shift_subj, t_grid, p_kappa, q
    )
    solved = _dp_layer_fixed_k(b_grid, best_cost, best_t, min_seg, k_seg=K_BMI)
    if solved is None:
        idx = np.linspace(0, len(b_grid), K_BMI + 1, dtype=int)
        idx[0], idx[-1] = 0, len(b_grid)
        times = []
        intervals = []
        for s in range(K_BMI):
            i_start, i_end = int(idx[s]), int(idx[s + 1])
            mask = (b_subj >= b_grid[i_start]) & (b_subj <= b_grid[i_end - 1])
            mask = mask if np.any(mask) else np.ones_like(b_subj, dtype=bool)
            t_opt = _best_time_for_segment(t_grid, b_subj[mask], shift_subj[mask], p_kappa, q)
            times.append(t_opt)
            intervals.append((float(b_grid[i_start]), float(b_grid[i_end - 1])))
        return intervals, enforce_monotonicity_q3(times)

    breaks, times, intervals = solved
    # Apply q2-style conservative floor per segment after the DP solution is found.
    for s in range(K_BMI):
        i_start, i_end = int(breaks[s]), int(breaks[s + 1])
        t_floor = float(np.median(t_floor_curve[i_start:i_end]))
        times[s] = max(times[s], t_floor)
    times = enforce_monotonicity_q3(times)
    return intervals, times


def enforce_age_monotonicity(t_matrix):
    """Column-wise isotonic smoothing so older layers are not earlier, without hard staircase."""
    t = np.asarray(t_matrix, dtype=float).copy()
    x = np.arange(t.shape[0], dtype=float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    # Process each BMI-segment column independently to preserve cross-segment structure.
    for j in range(t.shape[1]):
        t[:, j] = iso.fit_transform(x, t[:, j])
        for i in range(1, t.shape[0]):
            t[i, j] = max(t[i, j], t[i - 1, j] + AGE_MONO_MIN_INC)
    return np.clip(t, Q3_T_MIN, Q3_T_MAX)


def _cell_grid_counts(interval_matrix):
    """Weights by BMI grid counts per age-layer/BMI cell."""
    counts = np.zeros((K_AGE, K_BMI), dtype=float)
    for a in range(K_AGE):
        for s in range(K_BMI):
            b_lo, b_hi = interval_matrix[a][s]
            n_grid = int(np.floor((b_hi - b_lo) / BMI_STEP + 1e-9)) + 1
            counts[a, s] = float(max(1, n_grid))
    return counts


def weighted_avg_ga(t_matrix, cell_counts):
    # Use grid-count weights for the q3 summary average.
    w = np.asarray(cell_counts, dtype=float)
    t = np.asarray(t_matrix, dtype=float)
    if float(np.sum(w)) <= 0:
        return float(np.mean(t))
    return float(np.sum(w * t) / np.sum(w))


def solve_5x5(subjects, result, ga_bar, bmi_bar, q=0.95, kappa=1.0):
    """Solve 5x5 recommendations with age-layer specific BMI segmentation."""
    t_grid = get_t_grid(Q3_T_MAX)
    p_kappa, r_fail = make_p_kappa_and_r_fail(result, ga_bar, bmi_bar, kappa=kappa)

    bmi_min = float(subjects["bmi"].min())
    bmi_max = float(subjects["bmi"].max())
    b_grid = np.arange(bmi_min, bmi_max + 1e-9, BMI_STEP)
    b_grid = b_grid[b_grid <= bmi_max]
    # Estimate covariate offsets once per (q, kappa) setting before solving age-layer partitions.
    gamma = estimate_gamma_offset(subjects, b_grid, t_grid, p_kappa, r_fail, q)

    age_labels, age_mask_fn = fixed_age_layer_specs()
    delta_age = estimate_age_layer_offsets(subjects, p_kappa, q, t_grid)
    age_values = subjects["age"].to_numpy(dtype=float)

    t_matrix = []
    interval_matrix = []
    # Solve each age layer independently, then apply a cross-layer monotonic adjustment.
    for layer in range(K_AGE):
        mask_layer = age_mask_fn(age_values, layer)
        subjects_k = subjects.loc[mask_layer].copy()
        intervals, layer_times = _solve_one_age_layer(
            subjects_k, p_kappa, q, gamma, delta_age[layer], t_grid
        )
        t_matrix.append(layer_times)
        interval_matrix.append(intervals)

    t_matrix = enforce_age_monotonicity(t_matrix)
    cell_counts = _cell_grid_counts(interval_matrix)
    diagnostics = {
        "delta_age": [float(x) for x in delta_age],
    }
    return age_labels, np.asarray(t_matrix, dtype=float), interval_matrix, gamma, diagnostics, cell_counts


def print_table7(age_labels, t_matrix):
    age_w = 12
    col_w = 12
    print("\nRecommended GA (wk) by age-layer x BMI-segment-rank")
    print("Segment BMI intervals are layer-specific, shown in the next table.")
    print(f"{'Age layer':<{age_w}}" + "".join(f"{'Seg' + str(s + 1):<{col_w}}" for s in range(K_BMI)))
    print("-" * (age_w + col_w * K_BMI))
    for a in range(K_AGE):
        vals = "".join(f"{t_matrix[a, b]:<{col_w}.1f}" for b in range(K_BMI))
        print(f"{age_labels[a]:<{age_w}}{vals}")


def print_layer_intervals(age_labels, interval_matrix, t_matrix):
    age_w = 12
    seg_w = 6
    bmi_w = 22
    ga_w = 12
    print("\nLayer-specific BMI intervals and recommended GA")
    print(f"{'Age layer':<{age_w}}{'Seg':<{seg_w}}{'BMI interval':<{bmi_w}}{'Rec. GA (wk)':>{ga_w}}")
    print("-" * (age_w + seg_w + bmi_w + ga_w))
    for a in range(K_AGE):
        age_label = age_labels[a]
        for s in range(K_BMI):
            b_lo, b_hi = interval_matrix[a][s]
            bmi_text = f"[{b_lo:.2f}, {b_hi:.2f}]"
            print(f"{age_label:<{age_w}}{s + 1:<{seg_w}}{bmi_text:<{bmi_w}}{t_matrix[a, s]:>{ga_w}.1f}")


def detection_error_analysis(result, ga_bar, bmi_bar, subjects, q_used, base_t=None, base_w=None):
    """Kappa sensitivity: report weighted average recommended GA and delta vs kappa=1.0."""
    kappa_values = [0.75, 1.0, 1.25, 1.5, 2.0]
    out = {}
    weights = {}
    # Allow reuse of the base solution to avoid recomputing the kappa=1.0 case.
    for kappa in kappa_values:
        if kappa == 1.0 and base_t is not None and base_w is not None:
            out[kappa] = np.asarray(base_t, dtype=float)
            weights[kappa] = np.asarray(base_w, dtype=float)
            continue
        _, t_mat, _, _, _, cell_counts = solve_5x5(subjects, result, ga_bar, bmi_bar, q=q_used, kappa=kappa)
        out[kappa] = t_mat
        weights[kappa] = cell_counts

    baseline_mean = weighted_avg_ga(out[1.0], weights[1.0])
    print("\nDetection error / kappa sensitivity")
    print(f"{'kappa':<8}{'Wtd Avg Rec. GA (wk)':<22}{'delta_t (wk)':<14}")
    print("-" * 44)
    for kappa in kappa_values:
        avg_ga = weighted_avg_ga(out[kappa], weights[kappa])
        print(f"{kappa:<8}{avg_ga:<22.2f}{(avg_ga - baseline_mean):+,.2f}")


def main():
    df = q1_lmm.load_and_clean()
    q1_lmm.set_logit_response(df)
    result, _, _ = q1_lmm.fit_full_model(df)
    ga_bar = float(df[COL_GA].mean())
    bmi_bar = float(df[COL_BMI].mean())
    subjects = load_subjects_with_covariates()

    # Keep q=0.95 as the primary report setting and print q=0.99 as sensitivity only.
    q_main = 0.95
    age_labels, t_matrix, interval_matrix, gamma, diagnostics, cell_counts = solve_5x5(
        subjects, result, ga_bar, bmi_bar, q=q_main, kappa=1.0
    )
    print(
        f"\nCovariate offset (gamma_AGE, gamma_G, gamma_P): "
        f"({gamma[0]:.4f}, {gamma[1]:.4f}, {gamma[2]:.4f})"
    )
    print(f"Age-layer baseline delta (wk): {diagnostics['delta_age']}")
    print(f"Weighted average recommended GA (wk): {weighted_avg_ga(t_matrix, cell_counts):.2f}")
    print_table7(age_labels, t_matrix)
    print_layer_intervals(age_labels, interval_matrix, t_matrix)
    detection_error_analysis(result, ga_bar, bmi_bar, subjects, q_main, base_t=t_matrix, base_w=cell_counts)
    print("\nq sensitivity (kappa=1.0)")
    print(f"{'q':<8}{'Wtd Avg Rec. GA (wk)':<22}{'delta_t vs q=0.95':<18}")
    print("-" * 48)
    t95, w95 = t_matrix, cell_counts
    # Resolve q=0.99 once and compare against the cached q=0.95 baseline summary.
    _, t99, _, _, _, w99 = solve_5x5(subjects, result, ga_bar, bmi_bar, q=0.99, kappa=1.0)
    avg95 = weighted_avg_ga(t95, w95)
    avg99 = weighted_avg_ga(t99, w99)
    print(f"{0.95:<8}{avg95:<22.2f}{0.0:+.2f}")
    print(f"{0.99:<8}{avg99:<22.2f}{(avg99 - avg95):+.2f}")


if __name__ == "__main__":
    main()

