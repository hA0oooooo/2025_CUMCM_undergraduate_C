"""
Q2: NIPT timing optimization by BMI groups.

Reference: lambda search for K (lam_hi 0.30), grid [10,28], conservative correction
T_final = max(T_bayes, T*), executable domain [12, 28]. LMM for p_kappa, DP + penalty.
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
# Search grid [10, 28] (reference --tmin 10); executable domain [12, 28]
T_GRID_MIN = 10.0
T_MIN, T_MAX = 12.0, 28.0
# Time risk: phi(t<=12)=0; 12<t<28 -> alpha*(t-12); t>=28 -> 16*alpha+beta*(t-28)
ALPHA, BETA = 0.05, 0.1
BMI_STEP = 0.1
T_STEP = 0.1
MIN_SEG_LEN = 2


def make_p_kappa(result, ga_bar, bmi_bar, kappa=1.0):
    """Compute coverage probability p_kappa(t,b) = P(FF >= theta|t,b). Reference typical: sd = sqrt(scale) * kappa."""
    beta = result.fe_params.values
    scale = result.scale
    y_thresh = np.log(THETA_FF / (1.0 - THETA_FF))
    sd_eta = np.sqrt(scale) * kappa

    def x_row(ga, bmi):
        ga_c = ga - ga_bar
        bmi_c = bmi - bmi_bar
        return np.array([1.0, ga_c, ga_c ** 2, bmi_c, bmi_c ** 2])

    def p_kappa(t, b):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        b_arr = np.atleast_1d(np.asarray(b, dtype=float))
        t_arr, b_arr = np.broadcast_arrays(t_arr, b_arr)
        out = np.empty(t_arr.size)
        for idx in range(t_arr.size):
            tt, bb = t_arr.flat[idx], b_arr.flat[idx]
            x = x_row(tt, bb).reshape(1, -1)
            eta = (x @ beta)[0]
            out.flat[idx] = 1.0 - stats.norm.cdf((y_thresh - eta) / sd_eta)
        return out.reshape(t_arr.shape) if out.size > 1 else float(out.flat[0])

    return p_kappa


def make_p_kappa_and_r_fail(result, ga_bar, bmi_bar, kappa=1.0):
    """Return (p_kappa, r_fail) for q3; r_fail = 1 - p_kappa."""
    p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa)
    def r_fail(t, b):
        return 1.0 - p_kappa(t, b)
    return p_kappa, r_fail


def phi_time_risk(t):
    """Time-dependent risk function: phi(t<=12)=0; 12<t<28 -> alpha*(t-12); t>=28 -> 16*alpha + beta*(t-28)."""
    t = np.asarray(t, dtype=float)
    out = np.where(t <= 12, 0.0, np.where(t < 28, ALPHA * (t - 12), 16 * ALPHA + BETA * (t - 28)))
    return out


def get_t_grid(t_max):
    """Generate candidate gestational age grid from T_GRID_MIN to t_max (reference --tmin 10)."""
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
    n = len(b_grid)
    n_t = len(t_grid)
    P = np.zeros((n_t, n))
    for ti in range(n_t):
        for bi in range(n):
            P[ti, bi] = p_kappa(t_grid[ti], b_grid[bi])
    T_star = np.empty(n)
    for bi in range(n):
        idx = next((i for i in range(n_t) if P[i, bi] >= q), n_t - 1)
        T_star[bi] = t_grid[idx]
    T_curve = np.maximum.accumulate(T_star)
    return T_curve, T_star, P


def compute_T_bayes_conservative(b_grid, T_star_raw, T_curve):
    """Reference: T_final(b) = max(T_bayes(b), T*(b)) so backtest >= q. Isotonic fit to raw T*."""
    iso = IsotonicRegression(out_of_bounds="clip")
    T_bayes = iso.fit_transform(b_grid, T_star_raw)
    T_final = np.maximum(T_bayes, T_curve)
    return T_final


def apply_conservative_correction(b_grid, breakpoints, segment_times, T_final):
    """Per segment: t_s = max(t_s, median T_final in segment). Reference conservative correction."""
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
    
    For each segment, finds time t that minimizes cost = Σ_{k=i}^j w(k)[r_fail(t,b(k)) + φ(t)].
    Uses precomputed P matrix for efficient computation.
    
    IMPORTANT: Only considers time points t that satisfy the coverage constraint:
    min_{k∈[i:j]} P_K(t, b(k)) ≥ q (constraint from Step2 in reference).
    """
    n = len(b_grid)
    n_t = len(t_grid)
    _, _, P = compute_T_star_curve(b_grid, t_grid, p_kappa, q)
    best_cost = np.full((n + 1, n + 1), np.inf)
    best_t = np.full((n + 1, n + 1), np.nan)
    
    phi_t = phi_time_risk(t_grid)
    for i in range(n):
        for j in range(i + 1, n + 1):
            w_seg = w[i:j]
            w_sum = float(np.sum(w_seg))
            cost_vec = np.full(n_t, np.inf)  # Initialize with inf for invalid time points
            for ti in range(n_t):
                p_seg = P[ti, i:j]
                # Check coverage constraint: min_{k∈[i:j]} P_K(t, b(k)) ≥ q
                min_p = float(np.min(p_seg))
                if min_p < q:
                    continue  # Skip this time point if constraint not satisfied
                sum_w_p = float(np.sum(w_seg * p_seg))
                cost_vec[ti] = w_sum - sum_w_p + w_sum * phi_t[ti]
            if np.any(np.isfinite(cost_vec)):
                ti_best = np.argmin(cost_vec)
                best_cost[i, j] = cost_vec[ti_best]
                best_t[i, j] = float(t_grid[ti_best])
    return best_cost, best_t


def dp_solve(n, K, best_cost, best_t):
    """Dynamic programming to find optimal K-segment partition of BMI range.
    
    DP[j,s] = minimum cost to partition first j points into s segments.
    DP[j,s] = min_{i} [DP[i,s-1] + best_cost[i+1,j]] with monotonicity constraint.
    """
    min_seg = max(MIN_SEG_LEN, n // 15)
    DP = np.full((n + 1, K + 1), np.inf)
    DP[0, 0] = 0.0
    parent = np.full((n + 1, K + 1), -1, dtype=int)
    last_t = np.full((n + 1, K + 1), np.nan)

    for j in range(min_seg, n + 1):
        DP[j, 1] = best_cost[0, j]
        parent[j, 1] = -1
        last_t[j, 1] = best_t[0, j]

    for s in range(2, K + 1):
        for j in range(s * min_seg, n + 1):
            best_val = np.inf
            best_i = -1
            for i in range((s - 1) * min_seg, j - min_seg):
                if not np.isfinite(DP[i, s - 1]):
                    continue
                t_prev = last_t[i, s - 1]
                t_curr = best_t[i + 1, j]
                if t_curr < t_prev:
                    continue
                cand = DP[i, s - 1] + best_cost[i + 1, j]
                if cand < best_val:
                    best_val = cand
                    best_i = i
            if best_i >= 0:
                DP[j, s] = best_val
                parent[j, s] = best_i
                last_t[j, s] = best_t[best_i + 1, j]

    def backtrack(j_final, s_final):
        """Reconstruct segment breakpoints from DP parent pointers."""
        bp_list = []
        j = j_final
        for s in range(s_final, 0, -1):
            i = parent[j, s]
            if i == -1:
                break
            bp_list.append(i + 1)
            j = i
        bp_list.reverse()
        breakpoints = [0] + bp_list + [j_final]
        result = [int(breakpoints[0])]
        for bp in breakpoints[1:]:
            if int(bp) != result[-1]:
                result.append(int(bp))
        return result

    breakpoints = backtrack(n, K)
    while len(breakpoints) < K + 1:
        breakpoints.append(n)

    segment_times = []
    for idx in range(K):
        i_start, i_end = int(breakpoints[idx]), int(breakpoints[idx + 1])
        segment_times.append(float(best_t[i_start, i_end]))
    segment_times = enforce_monotonicity(segment_times)
    return float(DP[n, K]), breakpoints, segment_times


def dp_solve_with_penalty(n, best_cost, best_t, lam):
    """DP with segment-count penalty: minimize sum of segment costs + lam * K over variable K.
    Returns (total_cost, breakpoints, segment_times, K)."""
    min_seg = max(MIN_SEG_LEN, n // 15)
    F = np.full(n + 1, np.inf)
    F[0] = 0.0
    parent = np.full(n + 1, -1, dtype=int)

    for j in range(min_seg, n + 1):
        for i in range(0, j - min_seg + 1):
            if not np.isfinite(F[i]):
                continue
            cand = F[i] + best_cost[i, j] + lam
            if cand < F[j]:
                F[j] = cand
                parent[j] = i

    if not np.isfinite(F[n]):
        return np.inf, [0, n], [float(T_MIN)], 1

    breakpoints = []
    j = n
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
    best_cost, best_t = precompute_costs(b_grid, t_grid, w, p_kappa, q)
    best_lam = lam_lo
    best_bp, best_times, best_K = None, None, 0

    for _ in range(60):
        lam_mid = (lam_lo + lam_hi) * 0.5
        _, breakpoints, segment_times, K = dp_solve_with_penalty(n, best_cost, best_t, lam_mid)
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
    n = len(b_grid)
    for s in range(K):
        i_start, i_end = int(breakpoints[s]), int(breakpoints[s + 1])
        if i_end <= i_start:
            continue
        b_seg = b_grid[i_start:i_end]
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
    """Reference flow: lambda search for K, then conservative correction T_final (median per segment), monotonicity. Grid [10,28], output in [12,28]."""
    T_curve, T_star, _ = compute_T_star_curve(b_grid, t_grid, p_kappa, q)
    T_final = compute_T_bayes_conservative(b_grid, T_star, T_curve)
    _, breakpoints, segment_times, _ = find_lambda_for_k(
        b_grid, t_grid, w, p_kappa, q, target_k=target_k, lam_lo=0.0, lam_hi=LAM_HI
    )
    segment_times = apply_conservative_correction(b_grid, breakpoints, segment_times, T_final)
    stats_list = segment_stats(b_grid, breakpoints, segment_times, p_kappa, w)
    return breakpoints, segment_times, stats_list


def print_results(stats_list, title="Q2 NIPT groups"):
    """Print table: group, BMI interval, recommended GA, mean coverage, mean risk."""
    print(f"\n{title}")
    print(f"{'Group':<8}{'BMI interval':<24}{'Rec. GA (wk)':<14}{'Mean coverage':<16}{'Mean risk':<12}")
    print("-" * 74)
    for s, row in enumerate(stats_list, 1):
        print(f"{s:<8}({row['b_lo']:.1f}, {row['b_hi']:.1f}]{'':<10}{row['t_s']:<14.1f}{row['mean_coverage']:<16.4f}{row['mean_risk']:<12.4f}")


def sensitivity_analysis(result, ga_bar, bmi_bar, b_grid, w):
    """Sensitivity analysis: vary q (0.95, 0.98) and kappa (0.75, 1.0, 1.25, 1.5).
    
    For each combination, solve for optimal segments and compute average recommended GA.
    Report Δt relative to baseline (kappa=1.0, q=0.95).
    """
    t_grid = get_t_grid(T_MAX)
    q_values = [0.95, 0.98]
    kappa_values = [0.75, 1.0, 1.25, 1.5]
    
    baseline_q = 0.95
    baseline_kappa = 1.0
    p_kappa_baseline = make_p_kappa(result, ga_bar, bmi_bar, kappa=baseline_kappa)
    _, segment_times_baseline, stats_list_baseline = run_one_solve(
        b_grid, t_grid, w, p_kappa_baseline, baseline_q
    )
    ts_baseline = [row["t_s"] for row in stats_list_baseline]
    baseline_avg_ga = float(np.average(ts_baseline))
    
    print("\nSensitivity (q and κ)")
    print(f"{'q':<8}{'κ':<8}{'Segments':<10}{'Avg Rec. GA (wk)':<18}{'Δt (wk)':<12}")
    print("-" * 56)
    
    for q in q_values:
        for kappa in kappa_values:
            p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa=kappa)
            _, segment_times, stats_list = run_one_solve(b_grid, t_grid, w, p_kappa, q)
            ts = [row["t_s"] for row in stats_list]
            avg_ga = float(np.average(ts))
            delta_t = avg_ga - baseline_avg_ga
            print(f"{q:<8}{kappa:<8}{len(ts):<10}{avg_ga:<18.2f}{delta_t:+.2f}")


def main():
    df = q1_lmm.load_and_clean()
    q1_lmm.set_logit_response(df)
    result, _, _ = q1_lmm.fit_full_model(df)
    ga_bar = float(df[COL_GA].mean())
    bmi_bar = float(df[COL_BMI].mean())
    bmi_min = float(df[COL_BMI].min())
    bmi_max = float(df[COL_BMI].max())
    q_main = 0.95
    b_grid, t_grid = discretize_bmi_t(bmi_min, bmi_max)
    n = len(b_grid)
    w = np.ones(n) / n
    p_kappa = make_p_kappa(result, ga_bar, bmi_bar, kappa=1.0)
    breakpoints, segment_times, stats_list = run_one_solve(
        b_grid, t_grid, w, p_kappa, q_main
    )
    print_results(stats_list, f"NIPT groups (q={q_main}, κ=1)")
    sensitivity_analysis(result, ga_bar, bmi_bar, b_grid, w)


if __name__ == "__main__":
    main()
