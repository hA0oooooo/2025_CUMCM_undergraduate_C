"""
Q1: mixed-effects model for FF with GA and BMI.

summary:
1) Fit a mixed-effects model on logit(FF) with GA/BMI fixed effects and subject random effects.
2) Report coefficients and model diagnostics.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boy.csv"

COL_GA = "检测孕周"
COL_BMI = "孕妇BMI"
COL_FF = "Y染色体浓度"
COL_SUBJECT = "孕妇代码"


def load_and_clean():
    """Load raw data, keep core columns, and build centered polynomial terms."""
    # Keep only the columns used by the mixed model to make downstream assumptions explicit.
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    df = df[[COL_SUBJECT, COL_GA, COL_BMI, COL_FF]].copy()
    df = df.dropna(subset=[COL_GA, COL_BMI, COL_FF])
    df = df.astype({COL_GA: float, COL_BMI: float, COL_FF: float})
    # FF is a ratio in (0,1). Clip before logit transform to avoid inf values.
    df[COL_FF] = np.clip(df[COL_FF], 1e-6, 1.0 - 1e-6)

    ga_bar = float(df[COL_GA].mean())
    bmi_bar = float(df[COL_BMI].mean())
    # Centering reduces collinearity when adding quadratic terms to the fixed-effects design.
    df["GA_c"] = df[COL_GA] - ga_bar
    df["BMI_c"] = df[COL_BMI] - bmi_bar
    df["GA_c2"] = df["GA_c"] ** 2
    df["BMI_c2"] = df["BMI_c"] ** 2
    return df


def set_logit_response(df):
    """Set y = logit(FF)."""
    p = np.asarray(df[COL_FF].values, dtype=float)
    df["y"] = np.log(p / (1.0 - p))


def build_design(df):
    """Build fixed/random design matrices used by statsmodels MixedLM."""
    # Match the fixed-effects specification used in the modeling report.
    exog = pd.DataFrame(
        {
            "const": 1.0,
            "GA_c": df["GA_c"],
            "GA_c2": df["GA_c2"],
            "BMI_c": df["BMI_c"],
            "BMI_c2": df["BMI_c2"],
        }
    )
    # Random effects include subject intercept and subject-specific GA slope.
    exog_re = np.column_stack([np.ones(len(df)), df["GA_c"].values])
    return exog, exog_re


def print_correlation(df):
    """Print Pearson/Spearman correlation of FF with GA and BMI."""
    ff = df[COL_FF].values
    ga = df[COL_GA].values
    bmi = df[COL_BMI].values
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
    """Fit mixed model with random intercept + GA_c slope using ML."""
    exog, exog_re = build_design(df)
    # Use subject ID as the grouping key so repeated tests share the same random effects.
    model = MixedLM(df["y"], exog, groups=df[COL_SUBJECT], exog_re=exog_re)
    with warnings.catch_warnings():
        # Silence common optimization warnings to keep console output focused on model results.
        warnings.filterwarnings("ignore", message=".*boundary.*")
        warnings.filterwarnings("ignore", message=".*[Cc]onverge.*")
        warnings.filterwarnings("ignore", message=".*[Rr]etrying.*")
        # Use ML to keep a single, deterministic estimation setup in this project.
        result = model.fit(reml=False)
    return result, exog, exog_re


def conditional_r2(result, exog, exog_re):
    """Conditional R2 on logit scale."""
    beta = result.fe_params.values
    var_fixed = float(np.var(exog.values @ beta))
    z = np.asarray(exog_re)
    cov_re = np.asarray(result.cov_re)
    # Compute the average random-effect variance contribution across observations.
    var_random = float(np.mean(np.sum(z * (z @ cov_re), axis=1)))
    sigma2_e = float(result.scale)
    var_total = var_fixed + var_random + sigma2_e
    return (var_fixed + var_random) / var_total


def r2_ff(result, df, exog, exog_re):
    """R2 on FF scale with fixed effects + BLUP random effects."""
    ff_obs = np.asarray(df[COL_FF].values, dtype=float)
    eta = np.asarray(exog.values @ result.fe_params.values)
    z = np.asarray(exog_re)
    groups = df[COL_SUBJECT].values
    re_dict = result.random_effects
    # Add subject-level BLUP random effects to build fitted values on the logit scale.
    for i, g in enumerate(groups):
        eta[i] += float(np.dot(z[i], re_dict[g]))
    ff_pred = np.exp(eta) / (1.0 + np.exp(eta))
    ss_res = float(np.sum((ff_obs - ff_pred) ** 2))
    ss_tot = float(np.sum((ff_obs - np.mean(ff_obs)) ** 2))
    return 1.0 - ss_res / ss_tot


def icc(result, ga_c_values):
    """Compute ICC from random-intercept/random-slope variance components."""
    cov_re = np.atleast_2d(result.cov_re)
    sigma2_u = float(cov_re[0, 0])
    sigma2_b = float(cov_re[1, 1])
    var_ga_c = float(np.var(ga_c_values))
    sigma2_e = float(result.scale)
    # Approximate the average within-subject correlation by integrating the random slope over observed GA spread.
    num = sigma2_u + sigma2_b * var_ga_c
    return num / (num + sigma2_e)


def ame_ga(result, ff_values):
    """Average marginal effect of GA on FF scale."""
    beta = result.fe_params
    ff = np.asarray(ff_values, dtype=float)
    # Transform the GA coefficient from logit scale to FF scale using the logistic derivative.
    return float(np.mean(beta["GA_c"] * ff * (1.0 - ff)))


def main():
    df = load_and_clean()
    print_correlation(df)
    set_logit_response(df)
    print("\nTransform: logit(FF); covariates: GA_c, BMI_c")

    # Fit once and reuse the same result object for all reported statistics.
    result, exog, exog_re = fit_full_model(df)

    fe = result.fe_params
    se = result.bse[: len(fe)]
    pvals = result.pvalues[: len(fe)]
    w_term, w_est, w_se, w_p = 10, 14, 12, 10
    print("\nFixed effects")
    print(f"{'term':<{w_term}}{'estimate':>{w_est}}{'SE':>{w_se}}{'p (Wald)':>{w_p}}")
    for t in fe.index:
        print(f"{t:<{w_term}}{fe[t]:>{w_est}.6f}{se[t]:>{w_se}.6f}{pvals[t]:>{w_p}.4f}")

    # Statsmodels may return covariance in a matrix-like object, so normalize to ndarray for safe indexing.
    cov_re = np.atleast_2d(result.cov_re)
    sigma2_e = float(result.scale)
    print("\nRandom effects (variance components)")
    print(f"  Random intercept:    Var = {cov_re[0, 0]:.6f},  SD = {np.sqrt(cov_re[0, 0]):.6f}")
    print(f"  Random slope (GA_c): Var = {cov_re[1, 1]:.6f},  SD = {np.sqrt(cov_re[1, 1]):.6f}")
    print(f"  Cov(intercept, GA_c): {cov_re[0, 1]:.6f}")
    print(f"  Residual:            Var = {sigma2_e:.6f},  SD = {np.sqrt(sigma2_e):.6f}")

    r2 = conditional_r2(result, exog, exog_re)
    r2_ff_val = r2_ff(result, df, exog, exog_re)
    icc_val = icc(result, df["GA_c"].values)
    ame = ame_ga(result, df[COL_FF].values)
    w_label = 32
    print("\n{:<{w}}{:.4f}".format("R2 (conditional, logit scale)", r2, w=w_label))
    print("{:<{w}}{:.4f}".format("R2 (FF scale)", r2_ff_val, w=w_label))
    print("{:<{w}}{:.4f}".format("ICC", icc_val, w=w_label))
    print("{:<{w}}{:.6f}".format("AME_GA (FF scale)", ame, w=w_label))


if __name__ == "__main__":
    main()


