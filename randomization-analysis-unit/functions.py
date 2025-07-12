import scipy.stats as stsb
import polars as pl
import numpy as np
import statsmodels.formula.api as smf
from typing import Literal


def generate_dataset(
    num_units,
    num_obs,
    distr: Literal["geom", "pareto"] = "geom",
    alpha=50,
    beta=50,
    impact=0.0,
    random_seed=None,
):
    """
    Generate a dataset using scipy distributions with vectorized operations.

    Parameters:
    -----------
    num_units : int
        Number of units in the dataset
    alpha : float, default=2
        Alpha parameter for beta distribution (base success rate)
    beta : float, default=5
        Beta parameter for beta distribution (base success rate)
    geom_p : float, default=0.2
        Probability parameter for geometric distribution (number of observations per unit)
    impact : float, default=0.0
        Impact of treatment on success rate (added to base success rate for test group)
    random_seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pl.DataFrame
        DataFrame with columns: unit_id, base_success_rate, treatment_group, outcome
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate number of observations per unit (vectorized)
    if distr == "pareto":
        b = num_obs * 1.0 / (num_obs - num_units)
        num_obs_per_unit = (
            np.floor(stsb.pareto.rvs(b=b, size=num_units)).astype(int) + 1
        )  # Ensure at least 1 observation per unit
    else:
        geom_p = (
            num_units * 1.0 / num_obs
        )  # Adjusted to ensure total observations match
        num_obs_per_unit = stsb.geom.rvs(p=geom_p, size=num_units)

    # Generate base success rates per unit (vectorized)
    base_success_rates = stsb.beta.rvs(a=alpha, b=beta, size=num_units)

    # Assign units to test (1) or control (0) group with 50% probability
    treatment_groups = stsb.bernoulli.rvs(p=0.5, size=num_units)

    # Create arrays for the final dataset
    unit_ids = np.repeat(np.arange(num_units), num_obs_per_unit)
    repeated_success_rates = np.repeat(base_success_rates, num_obs_per_unit)
    repeated_treatment_groups = np.repeat(treatment_groups, num_obs_per_unit)

    # Assign randomization on observation level
    obs_treatment_groups = stsb.bernoulli.rvs(p=0.5, size=np.sum(num_obs_per_unit))

    # Apply impact to test group observations
    adjusted_success_rates = repeated_success_rates + (
        repeated_treatment_groups * impact
    )

    # Create obs level success rates based on randomization
    obs_success_rates = repeated_success_rates + (obs_treatment_groups * impact)

    # Ensure success rates stay within [0, 1] bounds
    adjusted_success_rates = np.clip(adjusted_success_rates, 0, 1)
    obs_success_rates = np.clip(obs_success_rates, 0, 1)

    # Generate outcomes using adjusted success rates (vectorized)
    outcomes = stsb.bernoulli.rvs(p=adjusted_success_rates)
    obs_outcomes = stsb.bernoulli.rvs(p=obs_success_rates)

    # Create DataFrame directly from arrays
    df = pl.DataFrame(
        {
            "unit_id": unit_ids,
            "base_success_rate": repeated_success_rates,
            "treatment_group": repeated_treatment_groups,
            "obs_treatment_groups": obs_treatment_groups,
            "adjusted_success_rate": adjusted_success_rates,
            "obs_success_rate": obs_success_rates,
            "outcome": outcomes,
            "obs_outcome": obs_outcomes,
        }
    )

    return df


def simple_ols_method(df):
    """Simple OLS regression method"""
    df_pandas = df.to_pandas()
    model = smf.ols("outcome ~ treatment_group", data=df_pandas).fit()
    return {
        "model": "simple_ols",
        "estimate": model.params["treatment_group"],
        "significant": int(model.pvalues["treatment_group"] < 0.05),
        "ci_lower": model.conf_int().loc["treatment_group", 0],
        "ci_upper": model.conf_int().loc["treatment_group", 1],
    }


def clustered_ols_method(df):
    """OLS with clustered standard errors method"""
    df_pandas = df.to_pandas()
    model = smf.ols("outcome ~ treatment_group", data=df_pandas).fit(
        cov_type="cluster", cov_kwds={"groups": df_pandas["unit_id"]}
    )
    return {
        "model": "clustered_ols",
        "estimate": model.params["treatment_group"],
        "significant": int(model.pvalues["treatment_group"] < 0.05),
        "ci_lower": model.conf_int().loc["treatment_group", 0],
        "ci_upper": model.conf_int().loc["treatment_group", 1],
    }


def unit_level_weighted_method(df):
    """Unit-level regression with weights method"""
    df_pandas = df.to_pandas()
    unit_level = (
        df_pandas.groupby("unit_id")
        .agg({"outcome": "mean", "treatment_group": "first", "obs_outcome": "size"})
        .reset_index()
    )
    model = smf.wls(
        "outcome ~ treatment_group", data=unit_level, weights=unit_level["obs_outcome"]
    ).fit()
    return {
        "model": "unit_level_ols_weighted",
        "estimate": model.params["treatment_group"],
        "significant": int(model.pvalues["treatment_group"] < 0.05),
        "ci_lower": model.conf_int().loc["treatment_group", 0],
        "ci_upper": model.conf_int().loc["treatment_group", 1],
    }


def unit_level_unweighted_method(df):
    """Unit-level regression without weights method"""
    df_pandas = df.to_pandas()
    unit_level = (
        df_pandas.groupby("unit_id")
        .agg({"outcome": "mean", "treatment_group": "first", "obs_outcome": "size"})
        .reset_index()
    )
    model = smf.ols("outcome ~ treatment_group", data=unit_level).fit()
    return {
        "model": "unit_level_ols_unweighted",
        "estimate": model.params["treatment_group"],
        "significant": int(model.pvalues["treatment_group"] < 0.05),
        "ci_lower": model.conf_int().loc["treatment_group", 0],
        "ci_upper": model.conf_int().loc["treatment_group", 1],
    }


def obs_level_randomization_method(df):
    """Observation-level randomization method"""
    df_pandas = df.to_pandas()
    model = smf.ols("obs_outcome ~ obs_treatment_groups", data=df_pandas).fit()
    return {
        "model": "obs_level_randomization",
        "estimate": model.params["obs_treatment_groups"],
        "significant": int(model.pvalues["obs_treatment_groups"] < 0.05),
        "ci_lower": model.conf_int().loc["obs_treatment_groups", 0],
        "ci_upper": model.conf_int().loc["obs_treatment_groups", 1],
    }


def bootstrap_method(df):
    """Bootstrap resampling method"""
    n_boot = 1000
    boot_estimates = []
    for _ in range(n_boot):
        random_assignments = (
            df.group_by("unit_id")
            .agg(pl.first("treatment_group").alias("treatment_group"))
            .with_columns(
                pl.col("treatment_group").sample(
                    fraction=1, with_replacement=True, shuffle=True, seed=42
                )
            )
        )
        treat_sample = (
            df.join(random_assignments, on="unit_id")
            .filter(pl.col("treatment_group") == 1)["outcome"]
            .to_numpy()
        )
        control_sample = (
            df.join(random_assignments, on="unit_id")
            .filter(pl.col("treatment_group") == 0)["outcome"]
            .to_numpy()
        )
        boot_estimates.append(treat_sample.mean() - control_sample.mean())
    boot_estimates = np.array(boot_estimates)
    boot_point = boot_estimates.mean()
    boot_ci_lower, boot_ci_upper = np.percentile(boot_estimates, [2.5, 97.5])
    return {
        "model": "bootstrap_ttest",
        "estimate": boot_point,
        "significant": int((boot_ci_lower > 0) or (boot_ci_upper < 0)),
        "ci_lower": boot_ci_lower,
        "ci_upper": boot_ci_upper,
    }


def bootstrap_method_optimized(df):
    """Optimized bootstrap resampling method - correctly resamples entire units"""
    # Convert to pandas once
    df_pandas = df.to_pandas()

    # Pre-process: create dictionaries mapping unit_id to outcomes for each group
    treatment_data = df_pandas[df_pandas["treatment_group"] == 1]
    control_data = df_pandas[df_pandas["treatment_group"] == 0]

    # Group outcomes by unit_id for each treatment group
    treatment_unit_outcomes = (
        treatment_data.groupby("unit_id")["outcome"].apply(list).to_dict()
    )
    control_unit_outcomes = (
        control_data.groupby("unit_id")["outcome"].apply(list).to_dict()
    )

    # Get unit IDs
    treatment_units = list(treatment_unit_outcomes.keys())
    control_units = list(control_unit_outcomes.keys())

    n_boot = 500
    rng = np.random.default_rng(42)

    boot_estimates = []
    for i in range(n_boot):
        # Resample unit IDs with replacement
        boot_treatment_unit_ids = rng.choice(
            treatment_units, size=len(treatment_units), replace=True
        )
        boot_control_unit_ids = rng.choice(
            control_units, size=len(control_units), replace=True
        )

        # Collect all outcomes from resampled units
        boot_treatment_outcomes = []
        for unit_id in boot_treatment_unit_ids:
            boot_treatment_outcomes.extend(treatment_unit_outcomes[unit_id])

        boot_control_outcomes = []
        for unit_id in boot_control_unit_ids:
            boot_control_outcomes.extend(control_unit_outcomes[unit_id])

        # Calculate difference in means
        treatment_mean = np.mean(boot_treatment_outcomes)
        control_mean = np.mean(boot_control_outcomes)
        boot_estimates.append(treatment_mean - control_mean)

    boot_estimates = np.array(boot_estimates)
    boot_point = boot_estimates.mean()
    boot_ci_lower, boot_ci_upper = np.percentile(boot_estimates, [2.5, 97.5])

    return {
        "model": "bootstrap_optimized",
        "estimate": boot_point,
        "significant": int((boot_ci_lower > 0) or (boot_ci_upper < 0)),
        "ci_lower": boot_ci_lower,
        "ci_upper": boot_ci_upper,
    }


def delta_method(df):
    """
    Delta method for variance estimation (Deng et al. 2011)

    Calculates variance for proportion metrics at observation level
    when randomization is at unit level.
    """
    df_pandas = df.to_pandas()

    # Aggregate data by unit_id and treatment_group
    unit_stats = (
        df_pandas.groupby(["unit_id", "treatment_group"])
        .agg(
            {
                "outcome": [
                    "sum",
                    "count",
                ]  # S_i (converted sessions), N_i (total sessions)
            }
        )
        .reset_index()
    )

    # Flatten column names
    unit_stats.columns = ["unit_id", "treatment_group", "S_i", "N_i"]

    # Split by treatment group
    treatment_stats = unit_stats[unit_stats["treatment_group"] == 1]
    control_stats = unit_stats[unit_stats["treatment_group"] == 0]

    # Treatment group calculations
    S_bar_T = treatment_stats["S_i"].mean()  # Mean converted sessions
    N_bar_T = treatment_stats["N_i"].mean()  # Mean total sessions
    Var_S_T = treatment_stats["S_i"].var(ddof=1)  # Variance in converted sessions
    Var_N_T = treatment_stats["N_i"].var(ddof=1)  # Variance in total sessions
    Cov_SN_T = np.cov(treatment_stats["S_i"], treatment_stats["N_i"])[
        0, 1
    ]  # Covariance
    n_T = len(treatment_stats)  # Number of treatment units

    # Control group calculations
    S_bar_C = control_stats["S_i"].mean()
    N_bar_C = control_stats["N_i"].mean()
    Var_S_C = control_stats["S_i"].var(ddof=1)
    Var_N_C = control_stats["N_i"].var(ddof=1)
    Cov_SN_C = np.cov(control_stats["S_i"], control_stats["N_i"])[0, 1]
    n_C = len(control_stats)

    # Delta method variance estimation
    # Var(p_T) = (1/(N_bar_T)^2) * Var(S_T) + (S_bar_T)^2/(N_bar_T)^4 * Var(N_T) - 2*S_bar_T/(N_bar_T)^3 * Cov(S_T, N_T)
    Var_p_T = (
        (1 / N_bar_T**2) * Var_S_T
        + (S_bar_T**2 / N_bar_T**4) * Var_N_T
        - (2 * S_bar_T / N_bar_T**3) * Cov_SN_T
    )

    Var_p_C = (
        (1 / N_bar_C**2) * Var_S_C
        + (S_bar_C**2 / N_bar_C**4) * Var_N_C
        - (2 * S_bar_C / N_bar_C**3) * Cov_SN_C
    )

    # Proportion estimates
    p_T = S_bar_T / N_bar_T
    p_C = S_bar_C / N_bar_C

    # Treatment effect estimate
    estimate = p_T - p_C

    # Standard error and Z-test
    se_delta = np.sqrt(Var_p_T / n_T + Var_p_C / n_C)
    z_stat = estimate / se_delta
    p_value = 2 * (1 - stsb.norm.cdf(abs(z_stat)))

    # 95% confidence interval
    ci_lower = estimate - 1.96 * se_delta
    ci_upper = estimate + 1.96 * se_delta

    return {
        "model": "delta_method",
        "estimate": estimate,
        "significant": int(p_value < 0.05),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def analyze_treatment_effect(df, methods=None):
    """
    Perform multiple statistical analyses to estimate treatment impact.

    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame with columns: unit_id, treatment_group, outcome
    methods : list of callables, optional
        List of statistical method functions to apply. If None, uses all available methods.

    Returns:
    --------
    list
        List of dictionaries with results for each method containing estimate, significant, ci_lower, ci_upper
    """
    # Default methods if none provided
    if methods is None:
        methods = [
            # simple_ols_method,
            clustered_ols_method,
            # unit_level_weighted_method,
            unit_level_unweighted_method,
            obs_level_randomization_method,
            bootstrap_method_optimized,
            delta_method,
        ]

    # Apply each method and collect results
    results = []
    for method in methods:
        results.append(method(df))

    return results
