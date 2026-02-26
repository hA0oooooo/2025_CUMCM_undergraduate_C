"""
Q2: NIPT timing optimization by BMI groups.

summary:
1) Convert the q1 mixed model into a coverage probability function p_kappa(t, b).
2) Solve the BMI-group timing schedule with penalized dynamic programming and conservative correction.
3) Report group recommendations and sensitivity to q and kappa.
"""
import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression

import q1_lmm

COL_GA = "检测孕周"
COL_BMI = "孕妇BMI"
THETA_FF = 0.04
K_SEG = 5
LAM_HI = 0.30
# Search grid [10, 28] (model grid lower bound 10); executable domain [12, 28]
T_GRID_MIN = 10.0
T_MIN, T_MAX = 12.0, 28.0
# Time risk: phi(t<=12)=0; 12<t<28 -> alpha*(t-12); t>=28 -> 16*alpha+beta*(t-28)
ALPHA, BETA = 0.05, 0.1
BMI_STEP = 0.1
T_STEP = 0.1
MIN_SEG_LEN = 2


def make_p_kappa(result, ga_bar, bmi_bar, kappa=1.0):
    """Compute coverage probability p_kappa(t,b) = P(FF >= theta|t,b) with sd = sqrt(scale) * kappa."""
    beta = result.fe_params.values
    scale = result.scale
    y_thresh = np.log(THETA_FF / (1.0 - THETA_FF))
    sd_eta = np.sqrt(scale) * kappa

    def p_kappa(t, b):
        t_arr = np.asarray(t, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        t_arr, b_arr = np.broadcast_arrays(t_arr, b_arr)
        ga_c = t_arr - ga_bar
        bmi_c = b_arr - bmi_bar
        eta = (
            beta[0]
            + beta[1] * ga_c
            + beta[2] * (ga_c ** 2)
            + beta[3] * bmi_c
            + beta[4] * (bmi_c ** 2)
        )
        z = (y_thresh - eta) / sd_eta
        return 1.0 - stats.norm.cdf(z)

    return p_kappa


def make_p_kappa_and_r_fail(result, ga_bar, bmi_bar, kappa=1.0):
    """Return (p_kappa, r_fail) for q3; r_fail = 1 - p_kappa."""
    p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa)
    # Expose failure risk as a callable so q3 can reuse the same calibrated coverage model.
    def r_fail(t, b):
        return 1.0 - p_kappa(t, b)
    return p_kappa, r_fail


def phi_time_risk(t):
    """Time-dependent risk function: phi(t<=12)=0; 12<t<28 -> alpha*(t-12); t>=28 -> 16*alpha + beta*(t-28)."""
    t = np.asarray(t, dtype=float)
    out = np.where(t <= 12, 0.0, np.where(t < 28, ALPHA * (t - 12), 16 * ALPHA + BETA * (t - 28)))
    return out


def get_t_grid(t_max):
    """Generate candidate gestational age grid from T_GRID_MIN to t_max."""
    t_grid = np.arange(T_GRID_MIN, t_max + 1e-9, T_STEP)
    return np.unique(t_grid)


def discretize_bmi_t(bmi_min, bmi_max):
    """Create discretized BMI and gestational age grids for dynamic programming."""
    b_grid = np.arange(bmi_min, bmi_max + 1e-9, BMI_STEP)
    b_grid = np.unique(b_grid)
    t_grid = get_t_grid(T_MAX)
    return b_grid, t_grid


def compute_T_star_curve(b_grid, t_grid, p_kappa, q):
    """For each BMI point b, compute T*(b) = earliest t such that p(t,b) >= q; else t_max.
    Return T_curve (monotonic), raw T_star, and P matrix."""
    t_grid = np.asarray(t_grid, dtype=float)
    b_grid = np.asarray(b_grid, dtype=float)
    P = p_kappa(t_grid[:, None], b_grid[None, :])
    mask = P >= q
    any_mask = mask.any(axis=0)
    # For each BMI grid point, pick the earliest time that satisfies the coverage threshold.
    idx = np.where(any_mask, mask.argmax(axis=0), len(t_grid) - 1)
    T_star = t_grid[idx]
    T_curve = np.maximum.accumulate(T_star)
    return T_curve, T_star, P


def compute_T_bayes_conservative(b_grid, T_star_raw, T_curve):
    """Use T_final(b) = max(T_bayes(b), T*(b)) so the schedule is not earlier than the threshold curve."""
    iso = IsotonicRegression(out_of_bounds="clip")
    # Isotonic regression smooths the raw threshold curve while preserving monotonicity.
    T_bayes = iso.fit_transform(b_grid, T_star_raw)
    T_final = np.maximum(T_bayes, T_curve)
    return T_final


def apply_conservative_correction(b_grid, breakpoints, segment_times, T_final):
    """Per segment, enforce t_s >= median T_final in that segment as a conservative correction."""
    K = len(segment_times)
    new_times = []
    for s in range(K):
        i_start, i_end = int(breakpoints[s]), int(breakpoints[s + 1])
        if i_end <= i_start:
            new_times.append(segment_times[s])
            continue
        t_floor = float(np.median(T_final[i_start:i_end]))
        new_times.append(max(segment_times[s], t_floor))
    return enforce_monotonicity(new_times)


def precompute_costs(b_grid, t_grid, w, p_kappa, q):
    """Precompute cost and recommended time for each segment [i:j).
    
    For each segment, finds time t that minimizes cost = 危_{k=i}^j w(k)[r_fail(t,b(k)) + 蠁(t)].
    Uses precomputed P matrix for efficient computation.
    
    IMPORTANT: Only considers time points t that satisfy the coverage constraint:
    min_{k鈭圼i:j]} P_K(t, b(k)) 鈮?q.
    """
    n = len(b_grid)
    _, _, P = compute_T_star_curve(b_grid, t_grid, p_kappa, q)
    best_cost = np.full((n + 1, n + 1), np.inf)
    best_t = np.full((n + 1, n + 1), np.nan)
    
    # Cache time-risk values once because they are reused for every candidate BMI segment.
    phi_t = phi_time_risk(t_grid)
    for i in range(n):
        for j in range(i + 1, n + 1):
            # Segment cost is evaluated on the BMI grid slice [i:j) with its local weights.
            w_seg = w[i:j]
            w_sum = float(np.sum(w_seg))
            p_seg = P[:, i:j]
            min_p_per_t = np.min(p_seg, axis=1)
            sum_w_p = p_seg @ w_seg
            # Cost = weighted failure risk plus weighted time risk over the current BMI segment.
            cost_vec = w_sum - sum_w_p + w_sum * phi_t
            # Discard candidate times that violate the hard coverage guarantee.
            cost_vec[min_p_per_t < q] = np.inf
            if np.any(np.isfinite(cost_vec)):
                ti_best = np.argmin(cost_vec)
                best_cost[i, j] = cost_vec[ti_best]
                best_t[i, j] = float(t_grid[ti_best])
    return best_cost, best_t


def dp_solve_with_penalty(n, best_cost, best_t, lam):
    """DP with segment-count penalty: minimize sum of segment costs + lam * K over variable K.
    Returns (total_cost, breakpoints, segment_times, K)."""
    # Keep a minimum segment width to prevent pathological tiny partitions on fine BMI grids.
    min_seg = max(MIN_SEG_LEN, n // 15)
    F = np.full(n + 1, np.inf)
    F[0] = 0.0
    parent = np.full(n + 1, -1, dtype=int)

    for j in range(min_seg, n + 1):
        for i in range(0, j - min_seg + 1):
            if not np.isfinite(F[i]):
                continue
            # Transition adds the segment cost and a penalty on the segment count.
            cand = F[i] + best_cost[i, j] + lam
            if cand < F[j]:
                F[j] = cand
                parent[j] = i

    if not np.isfinite(F[n]):
        return np.inf, [0, n], [float(T_MIN)], 1

    breakpoints = []
    j = n
    # Backtrack the optimal partition from the parent pointers.
    while j > 0:
        i = parent[j]
        breakpoints.append(j)
        j = i
    breakpoints.reverse()
    breakpoints = [0] + breakpoints
    K = len(breakpoints) - 1
    segment_times = []
    for idx in range(K):
        i_start, i_end = int(breakpoints[idx]), int(breakpoints[idx + 1])
        segment_times.append(float(best_t[i_start, i_end]))
    segment_times = enforce_monotonicity(segment_times)
    return float(F[n]), breakpoints, segment_times, K


def find_lambda_for_k(b_grid, t_grid, w, p_kappa, q, target_k, lam_lo=0.0, lam_hi=0.5, tol=1e-4):
    """Binary search lambda so that DP with penalty yields segment count close to target_k (e.g. 4 or 5).
    Precomputes costs once; returns (lam, breakpoints, segment_times, K)."""
    n = len(b_grid)
    # Cost tables are the expensive part, so compute them once before the lambda search.
    best_cost, best_t = precompute_costs(b_grid, t_grid, w, p_kappa, q)
    best_lam = lam_lo
    best_bp, best_times, best_K = None, None, 0

    for _ in range(60):
        lam_mid = (lam_lo + lam_hi) * 0.5
        _, breakpoints, segment_times, K = dp_solve_with_penalty(n, best_cost, best_t, lam_mid)
        # Binary search lambda so the penalized DP returns a segment count close to target_k.
        if K == target_k:
            return lam_mid, breakpoints, segment_times, K
        if K > target_k:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        best_lam = lam_mid
        best_bp, best_times, best_K = breakpoints, segment_times, K
        if lam_hi - lam_lo < tol:
            break
    return best_lam, best_bp, best_times, best_K


def enforce_monotonicity(t_list):
    """Enforce monotonicity: t_1 <= t_2 <= ... <= t_K, and clip all values to [T_MIN, T_MAX]."""
    t_arr = np.array(t_list, dtype=float)
    t_arr = np.clip(t_arr, T_MIN, T_MAX)
    for i in range(1, len(t_arr)):
        t_arr[i] = max(t_arr[i], t_arr[i - 1])
    return t_arr.tolist()


def segment_stats(b_grid, breakpoints, segment_times, p_kappa, w):
    """For each segment, compute mean coverage and mean risk (1 - p_kappa + phi)."""
    K = len(segment_times)
    stats_list = []
    for s in range(K):
        i_start, i_end = int(breakpoints[s]), int(breakpoints[s + 1])
        if i_end <= i_start:
            continue
        b_seg = b_grid[i_start:i_end]
        # Normalize local weights so segment metrics are comparable across segments.
        w_seg = w[i_start:i_end] / w[i_start:i_end].sum()
        t_s = segment_times[s]
        p_seg = p_kappa(t_s, b_seg)
        phi_s = phi_time_risk(t_s)
        mean_coverage = float(np.sum(w_seg * p_seg))
        mean_risk = float(np.sum(w_seg * (1.0 - p_seg + phi_s)))
        stats_list.append({
            "b_lo": float(b_grid[i_start]),
            "b_hi": float(b_grid[i_end - 1]),
            "t_s": t_s,
            "mean_coverage": mean_coverage,
            "mean_risk": mean_risk,
        })
    return stats_list


def run_one_solve(b_grid, t_grid, w, p_kappa, q, target_k=K_SEG):
    """Current flow: lambda search for K, conservative correction, then monotonicity. Grid [10,28], output in [12,28]."""
    T_curve, T_star, _ = compute_T_star_curve(b_grid, t_grid, p_kappa, q)
    T_final = compute_T_bayes_conservative(b_grid, T_star, T_curve)
    # The DP stage chooses a piecewise-constant schedule before conservative post-correction.
    _, breakpoints, segment_times, _ = find_lambda_for_k(
        b_grid, t_grid, w, p_kappa, q, target_k=target_k, lam_lo=0.0, lam_hi=LAM_HI
    )
    segment_times = apply_conservative_correction(b_grid, breakpoints, segment_times, T_final)
    stats_list = segment_stats(b_grid, breakpoints, segment_times, p_kappa, w)
    return breakpoints, segment_times, stats_list


def print_results(stats_list, title="Q2 NIPT groups"):
    """Print table: group, BMI interval, recommended GA, mean coverage, mean risk."""
    print(f"\n{title}")
    print(f"{'Group':<8}{'BMI interval':<22}{'Rec. GA (wk)':>14}{'Mean coverage':>16}{'Mean risk':>12}")
    print("-" * 72)
    for s, row in enumerate(stats_list, 1):
        bmi_interval = f"({row['b_lo']:.1f}, {row['b_hi']:.1f}]"
        print(
            f"{s:<8}{bmi_interval:<22}"
            f"{row['t_s']:>14.1f}{row['mean_coverage']:>16.4f}{row['mean_risk']:>12.4f}"
        )


def sensitivity_analysis(result, ga_bar, bmi_bar, b_grid, w):
    """Sensitivity analysis: vary q (0.95, 0.98) and kappa (0.75, 1.0, 1.25, 1.5).
    
    For each combination, solve for optimal segments and compute average recommended GA.
    Report delta_t relative to baseline (kappa=1.0, q=0.95).
    """
    t_grid = get_t_grid(T_MAX)
    q_values = [0.95, 0.98]
    kappa_values = [0.75, 1.0, 1.25, 1.5]
    
    # Use q=0.95 and kappa=1.0 as the baseline operating point for delta_t reporting.
    baseline_q = 0.95
    baseline_kappa = 1.0
    p_kappa_baseline = make_p_kappa(result, ga_bar, bmi_bar, kappa=baseline_kappa)
    _, _, stats_list_baseline = run_one_solve(
        b_grid, t_grid, w, p_kappa_baseline, baseline_q
    )
    ts_baseline = [row["t_s"] for row in stats_list_baseline]
    baseline_avg_ga = float(np.average(ts_baseline))
    
    print("\nSensitivity (q and kappa)")
    print(f"{'q':<8}{'kappa':<8}{'Segments':<10}{'Avg Rec. GA (wk)':>18}{'delta_t (wk)':>14}")
    print("-" * 56)
    
    for q in q_values:
        for kappa in kappa_values:
            # Resolve each setting independently so the comparison reflects both q and noise changes.
            p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa=kappa)
            _, _, stats_list = run_one_solve(b_grid, t_grid, w, p_kappa, q)
            ts = [row["t_s"] for row in stats_list]
            avg_ga = float(np.average(ts))
            delta_t = avg_ga - baseline_avg_ga
            print(f"{q:<8}{kappa:<8}{len(ts):<10}{avg_ga:>18.2f}{delta_t:>+14.2f}")


def main():
    df = q1_lmm.load_and_clean()
    q1_lmm.set_logit_response(df)
    # Reuse the q1 mixed model so q2 stays consistent with the fitted FF dynamics.
    result, _, _ = q1_lmm.fit_full_model(df)
    ga_bar = float(df[COL_GA].mean())
    bmi_bar = float(df[COL_BMI].mean())
    bmi_min = float(df[COL_BMI].min())
    bmi_max = float(df[COL_BMI].max())
    q_main = 0.95
    b_grid, t_grid = discretize_bmi_t(bmi_min, bmi_max)
    # Use uniform BMI-grid weights in the base optimization.
    w = np.ones(len(b_grid)) / len(b_grid)
    p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa=1.0)
    _, _, stats_list = run_one_solve(
        b_grid, t_grid, w, p_kappa, q_main
    )
    print_results(stats_list, f"NIPT groups (q={q_main}, kappa=1)")
    sensitivity_analysis(result, ga_bar, bmi_bar, b_grid, w)


if __name__ == "__main__":
    main()



