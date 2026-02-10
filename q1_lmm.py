"""
Q1: LMM per plan (reference 4.3–4.6). Response y = logit(FF); covariates GA_c, BMI_c (mean-centered).
Fixed: const, GA_c, GA_c^2, BMI_c, BMI_c^2; random: intercept + slope(GA_c). ML (reml=False, aligned with reference).
Output: fixed effects (term, estimate, SE, p Wald), variance components, conditional R², ICC, AME_GA.
No figures in implementation phase (plan: visualization in final stage).

REML vs ML (reml=False => we use ML):
  ML (maximum likelihood): maximizes the full likelihood of the data given fixed effects and variance
  components. Fixed-effect estimates can be biased when the number of groups is small because the
  variance is estimated from the same data used to estimate the mean.
  REML (restricted/residual maximum likelihood): maximizes the likelihood of linear combinations of
  the data that do not depend on the fixed effects (residual likelihood). REML gives less biased
  estimates of variance components in small samples and is often preferred for inference on random
  effects; fixed-effect estimates and SEs can differ slightly from ML. Reference implementation uses
  ML (reml=False); we follow that here.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boy.csv"
IMAGE_DIR = ROOT / "image"

COL_GA = "检测孕周"
COL_BMI = "孕妇BMI"
COL_FF = "Y染色体浓度"
COL_SUBJECT = "孕妇代码"


def load_and_clean():
    """Load boy data; drop missing on GA, BMI, FF; clip FF to (eps, 1-eps); center GA/BMI, add squared terms (plan 4.3.1)."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[[COL_SUBJECT, COL_GA, COL_BMI, COL_FF]].copy()
    df = df.dropna(subset=[COL_GA, COL_BMI, COL_FF])
    df = df.astype({COL_GA: float, COL_BMI: float, COL_FF: float})
    df[COL_FF] = np.clip(df[COL_FF], 1e-6, 1.0 - 1e-6)
    ga_bar = df[COL_GA].mean()
    bmi_bar = df[COL_BMI].mean()
    df["GA_c"] = df[COL_GA] - ga_bar
    df["BMI_c"] = df[COL_BMI] - bmi_bar
    df["GA_c2"] = df["GA_c"] ** 2
    df["BMI_c2"] = df["BMI_c"] ** 2
    return df


def set_logit_response(df):
    """Set df['y'] = logit(FF) for linear scale (plan 4.3.1 eq.1)."""
    p = np.asarray(df[COL_FF].values, dtype=float)
    df["y"] = np.log(p / (1.0 - p))


def build_design(df):
    """Design matrices: fixed const, GA_c, GA_c^2, BMI_c, BMI_c^2; random intercept + GA_c (plan 4.3.2 eq.3–6)."""
    exog = pd.DataFrame({
        "const": 1.0,
        "GA_c": df["GA_c"],
        "GA_c2": df["GA_c2"],
        "BMI_c": df["BMI_c"],
        "BMI_c2": df["BMI_c2"],
    })
    exog_re = np.column_stack([np.ones(len(df)), df["GA_c"].values])
    return exog, exog_re


def print_correlation(df):
    """Pearson and Spearman correlation of FF with GA and BMI."""
    ff = df[COL_FF].values
    ga, bmi = df[COL_GA].values, df[COL_BMI].values
    r_ga, p_ga = stats.pearsonr(ff, ga)
    r_bmi, p_bmi = stats.pearsonr(ff, bmi)
    rho_ga, p_rho_ga = stats.spearmanr(ff, ga)
    rho_bmi, p_rho_bmi = stats.spearmanr(ff, bmi)
    w_label, w_num, w_p, w_rho = 10, 10, 8, 14
    print("Correlation of FF with GA and BMI")
    print(f"{'':<{w_label}}{'Pearson_r':>{w_num}}{'p':>{w_p}}{'Spearman_rho':>{w_rho}}{'p':>{w_p}}")
    print(f"{'FF vs GA':<{w_label}}{r_ga:>{w_num}.4f}{p_ga:>{w_p}.4f}{rho_ga:>{w_rho}.4f}{p_rho_ga:>{w_p}.4f}")
    print(f"{'FF vs BMI':<{w_label}}{r_bmi:>{w_num}.4f}{p_bmi:>{w_p}.4f}{rho_bmi:>{w_rho}.4f}{p_rho_bmi:>{w_p}.4f}")


def fit_full_model(df):
    """LMM: y ~ fixed + random; ML (reml=False)"""
    exog, exog_re = build_design(df)
    model = MixedLM(df["y"], exog, groups=df[COL_SUBJECT], exog_re=exog_re)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*boundary.*")
        warnings.filterwarnings("ignore", message=".*[Cc]onverge.*")
        warnings.filterwarnings("ignore", message=".*[Rr]etrying.*")
        # ML: maximize full likelihood; REML would maximize residual likelihood for variance (less bias in small samples, reference uses ML)
        result = model.fit(reml=False)  
    return result, exog, exog_re


def conditional_r2(result, exog, exog_re):
    """Conditional R² on logit scale (plan 4.4)."""
    beta = result.fe_params.values
    var_fixed = np.var(exog.values @ beta)
    z = np.asarray(exog_re)
    cov_re = np.asarray(result.cov_re)
    var_random = np.mean(np.sum(z * (z @ cov_re), axis=1))
    sigma2_e = result.scale
    var_total = var_fixed + var_random + sigma2_e
    return (var_fixed + var_random) / var_total


def r2_ff(result, df, exog, exog_re):
    """R² on FF scale using fixed + BLUP (plan 4.3.3)."""
    ff_obs = np.asarray(df[COL_FF].values, dtype=float)
    eta_fixed = np.asarray(exog.values @ result.fe_params.values)
    re_dict = result.random_effects
    groups = df[COL_SUBJECT].values
    z = np.asarray(exog_re)
    eta = eta_fixed.copy()
    for i, g in enumerate(groups):
        eta[i] += np.dot(z[i], re_dict[g])
    ff_pred = np.exp(eta) / (1.0 + np.exp(eta))
    ss_res = np.sum((ff_obs - ff_pred) ** 2)
    ss_tot = np.sum((ff_obs - np.mean(ff_obs)) ** 2)
    return 1.0 - (ss_res / ss_tot)


def icc(result, exog_re, ga_c_values):
    """ICC per plan 4.4 eq.11: (sigma_u^2 + sigma_b^2*Var(GA_c)) / (same + sigma_e^2)."""
    cov_re = np.atleast_2d(result.cov_re)
    sigma2_u = float(cov_re[0, 0])
    sigma2_b = float(cov_re[1, 1])
    var_ga_c = np.var(ga_c_values)
    sigma2_e = result.scale
    num = sigma2_u + sigma2_b * var_ga_c
    return num / (num + sigma2_e)


def ame_ga(result, ff_values):
    """Average marginal effect of GA on FF scale (plan 4.4 eq.10)."""
    beta = result.fe_params
    ff = np.asarray(ff_values, dtype=float)
    return np.mean(beta["GA_c"] * ff * (1.0 - ff))


def _fixed_pred_ff(ga, bmi, beta, ga_bar, bmi_bar):
    """Fixed-effect predicted FF at (GA, BMI)."""
    ga_c = ga - ga_bar
    bmi_c = bmi - bmi_bar
    eta = beta["const"] + beta["GA_c"] * ga_c + beta["GA_c2"] * (ga_c ** 2) + beta["BMI_c"] * bmi_c + beta["BMI_c2"] * (bmi_c ** 2)
    return np.exp(eta) / (1.0 + np.exp(eta))


def draw_heatmap(df):
    """Mean FF by GA and BMI intervals. Not called in implementation phase (plan: figures in final stage)."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    ga, bmi, ff = df[COL_GA].values, df[COL_BMI].values, df[COL_FF].values
    ga_edges = np.arange(9, 29, 1)
    bmi_edges = np.arange(18, 46, 1.5)
    H, _, _ = np.histogram2d(ga, bmi, bins=[ga_edges, bmi_edges], weights=ff)
    W, _, _ = np.histogram2d(ga, bmi, bins=[ga_edges, bmi_edges])
    mean_ff = H / W
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mean_ff, aspect="auto", origin="lower", cmap="RdYlBu_r",
                   extent=[bmi_edges[0], bmi_edges[-1], ga_edges[0], ga_edges[-1]],
                   vmin=0, vmax=0.2)
    plt.colorbar(im, ax=ax, label="Mean FF")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Gestational age (weeks)")
    ax.set_title("Mean FF across GA and BMI intervals")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "1-1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_ff_vs_ga(df, result, ga_bar, bmi_bar):
    """FF vs GA at BMI P10/P50/P90, 95% CI. Not called in implementation phase."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    beta = result.fe_params
    cov_fe = result.cov_params().iloc[: len(beta), : len(beta)]
    ga_grid = np.linspace(df[COL_GA].min(), df[COL_GA].max(), 80)
    bmi_p10, bmi_p50, bmi_p90 = np.percentile(df[COL_BMI], [10, 50, 90])
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, bmi_val in [("BMI P10", bmi_p10), ("BMI P50", bmi_p50), ("BMI P90", bmi_p90)]:
        ga_c = ga_grid - ga_bar
        bmi_c = bmi_val - bmi_bar
        x_row = np.column_stack([np.ones_like(ga_grid), ga_c, ga_c ** 2, np.full_like(ga_grid, bmi_c), np.full_like(ga_grid, bmi_c ** 2)])
        eta = x_row @ beta.values
        se_eta = np.sqrt(np.einsum("ni,ij,nj->n", x_row, cov_fe.values, x_row))
        eta_lo, eta_hi = eta - 1.96 * se_eta, eta + 1.96 * se_eta
        ff_pred = np.exp(eta) / (1.0 + np.exp(eta))
        ff_lo = np.exp(eta_lo) / (1.0 + np.exp(eta_lo))
        ff_hi = np.exp(eta_hi) / (1.0 + np.exp(eta_hi))
        ax.plot(ga_grid, ff_pred * 100, label=label)
        ax.fill_between(ga_grid, ff_lo * 100, ff_hi * 100, alpha=0.2)
    ax.axhline(4, color="red", linestyle="--", label="4% threshold")
    ax.set_xlabel("Gestational age (weeks)")
    ax.set_ylabel("FF (%)")
    ax.set_title("Mixed model: FF ~ GA (by BMI)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 25)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "1-2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_ff_vs_bmi(df, result, ga_bar, bmi_bar):
    """FF vs BMI at GA P25/P50/P75, 95% CI. Not called in implementation phase."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    beta = result.fe_params
    cov_fe = result.cov_params().iloc[: len(beta), : len(beta)]
    bmi_grid = np.linspace(df[COL_BMI].min(), df[COL_BMI].max(), 80)
    ga_p25, ga_p50, ga_p75 = np.percentile(df[COL_GA], [25, 50, 75])
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, ga_val in [("GA P25", ga_p25), ("GA P50", ga_p50), ("GA P75", ga_p75)]:
        ga_c = np.full_like(bmi_grid, ga_val - ga_bar)
        bmi_c = bmi_grid - bmi_bar
        x_row = np.column_stack([np.ones_like(bmi_grid), ga_c, ga_c ** 2, bmi_c, bmi_c ** 2])
        eta = x_row @ beta.values
        se_eta = np.sqrt(np.einsum("ni,ij,nj->n", x_row, cov_fe.values, x_row))
        eta_lo, eta_hi = eta - 1.96 * se_eta, eta + 1.96 * se_eta
        ff_pred = np.exp(eta) / (1.0 + np.exp(eta))
        ff_lo = np.exp(eta_lo) / (1.0 + np.exp(eta_lo))
        ff_hi = np.exp(eta_hi) / (1.0 + np.exp(eta_hi))
        ax.plot(bmi_grid, ff_pred * 100, label=label)
        ax.fill_between(bmi_grid, ff_lo * 100, ff_hi * 100, alpha=0.2)
    ax.axhline(4, color="red", linestyle="--", label="4% threshold")
    ax.set_xlabel("BMI")
    ax.set_ylabel("FF (%)")
    ax.set_title("Mixed model: FF ~ BMI (by GA)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 25)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "1-3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_and_clean()
    print_correlation(df)

    set_logit_response(df)
    print("\nTransform: logit(FF); covariates: GA_c, BMI_c (mean-centered)")

    result, exog, exog_re = fit_full_model(df)

    fe = result.fe_params
    se = result.bse[: len(fe)]
    pvals = result.pvalues[: len(fe)]
    w_term, w_est, w_se, w_p = 10, 14, 12, 10
    print("\nFixed effects")
    print(f"{'term':<{w_term}}{'estimate':>{w_est}}{'SE':>{w_se}}{'p (Wald)':>{w_p}}")
    for t in fe.index:
        print(f"{t:<{w_term}}{fe[t]:>{w_est}.6f}{se[t]:>{w_se}.6f}{pvals[t]:>{w_p}.4f}")

    cov_re = np.atleast_2d(result.cov_re)
    sigma2_e = result.scale
    print("\nRandom effects (variance components)")
    print(f"  Random intercept:    Var = {cov_re[0, 0]:.6f},  SD = {np.sqrt(cov_re[0, 0]):.6f}")
    print(f"  Random slope (GA_c): Var = {cov_re[1, 1]:.6f},  SD = {np.sqrt(cov_re[1, 1]):.6f}")
    print(f"  Cov(intercept, GA_c): {cov_re[0, 1]:.6f}")
    print(f"  Residual:            Var = {sigma2_e:.6f},  SD = {np.sqrt(sigma2_e):.6f}")

    r2 = conditional_r2(result, exog, exog_re)
    r2_ff_val = r2_ff(result, df, exog, exog_re)
    icc_val = icc(result, exog_re, df["GA_c"].values)
    ame = ame_ga(result, df[COL_FF].values)
    w_label = 32
    print("\n{:<{w}}{:.4f}".format("R2 (conditional, logit scale)", r2, w=w_label))
    print("{:<{w}}{:.4f}".format("R2 (FF scale)", r2_ff_val, w=w_label))
    print("{:<{w}}{:.4f}".format("ICC", icc_val, w=w_label))
    print("{:<{w}}{:.6f}".format("AME_GA (FF scale)", ame, w=w_label))

    # Visualization deferred to final stage (plan: no figures in implementation phase).


if __name__ == "__main__":
    main()
