# =============================================================================
# Import necessary libraries
# =============================================================================
import pandas as pd
import numpy as np
import os
import ast
from TSC.time_series import generate, delete
from datetime import datetime
import sys
from TSC.data_imputataion import impute
import random
import time
import gc
import tracemalloc
from collections import defaultdict
from TSC.metrics import similarity

from scipy.interpolate import CubicSpline, PchipInterpolator
from statsmodels.tsa.statespace.structural import UnobservedComponents



# =============================================================================
# Set working directory
# =============================================================================
# base_working_directory = '/lu/topola/home/mariasad/MS'
base_working_directory = r'C:\Users\m-sad\OneDrive\Pulpit\data_imputation\data'

os.chdir(base_working_directory)

data_file_path = os.path.join(
    base_working_directory,
    'data_imputation.xlsx'
)

# =============================================================================
# Load Excel files
# =============================================================================
g_number = int(sys.argv[1])

data = pd.read_excel(data_file_path)
data = data[data['g_number'] == g_number]

# =============================================================================
# Define and create necessary directories
# =============================================================================
g_date = data['g_date'].iloc[0]

working_directory = os.path.join(
    base_working_directory,
    f"{g_date}_{g_number}"
)

input_dir = os.path.join(working_directory, 'input_data')
output_dir = os.path.join(working_directory, 'output_data')
log_dir = os.path.join(working_directory, 'logi')
trained_models_dir = os.path.join(working_directory, 'plots')


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


for directory in [input_dir, output_dir, log_dir, trained_models_dir]:
    ensure_directory(directory)

# =============================================================================
# Logging configuration
# =============================================================================
log_filename = f'cross_validation_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'

# =============================================================================
# Generating time_series data if not exists
# =============================================================================
random_seeds = eval(data.iloc[0]['d_random_seeds'])

g_exists = int(data['g_exists'].iloc[0])

if g_exists == 0:

    output_type = str(data['d_output_type'].iloc[0])
    shapes = ast.literal_eval(data['d_shapes'].iloc[0])
    length = int(data['d_length'].iloc[0])

    seed = random_seeds[0]

    noise_distribution = data['d_noise_distribution'].iloc[0]
    noise_max = data['d_noise_max'].iloc[0]

    parameters_list = eval(data['d_parameters_list'].iloc[0])

    # =============================================================================
    # Add noise dynamically
    # =============================================================================
    if pd.notna(noise_distribution) and pd.notna(noise_max):

        noise_max = float(noise_max)
        noise_distribution = str(noise_distribution)

        if noise_distribution == 'norm':
            noise_params = {'noise': ('norm', 0, noise_max)}

        elif noise_distribution == 'uniform':
            noise_params = {
                'noise': ('uniform', -noise_max, noise_max)
            }

        else:
            noise_params = {}

        for i, shape_params in enumerate(parameters_list):

            if (
                isinstance(shape_params[-1], dict)
                and 'noise' not in shape_params[-1]
            ):
                shape_params.append(noise_params)

            elif (
                isinstance(shape_params[-1], dict)
                and 'noise' in shape_params[-1]
            ):
                continue

            else:
                shape_params.append(noise_params)

    # =============================================================================
    # Generate time series
    # =============================================================================
    time_series, desc = generate.generate_time_series(
        output_type,
        shapes,
        parameters_list,
        length,
        seed
    )

    # =============================================================================
    # Save generated data
    # =============================================================================
    time_series_path = os.path.join(
        input_dir,
        'time_series.csv'
    )

    if isinstance(time_series, pd.DataFrame):
        time_series.to_csv(time_series_path, index=False)

    elif isinstance(time_series, np.ndarray):
        pd.DataFrame(time_series).to_csv(
            time_series_path,
            index=False
        )

    desc_path = os.path.join(input_dir, 'desc.csv')
    desc.to_csv(desc_path, index=False)

else:
    # =============================================================================
    # Load existing time series
    # =============================================================================
    time_series_path = os.path.join(
        input_dir,
        'time_series.csv'
    )

    time_series = pd.read_csv(time_series_path)
    
    
# =============================================================================
# Helper
# =============================================================================

def save_df(obj, path):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    elif isinstance(obj, np.ndarray):
        pd.DataFrame(obj).to_csv(path, index=False)


def profile_callable(func):
    """
    Execute one imputation method in two controlled passes:

    1. time pass: measure wall-clock execution time with tracemalloc disabled;
    2. memory pass: measure peak Python-traced memory with tracemalloc enabled.

    Separating the passes prevents tracemalloc overhead from biasing the
    execution-time comparison. Python's ``random`` state and NumPy's legacy RNG
    state are restored before the memory pass, and the post-time-pass states are
    restored afterwards so profiling does not consume the random sequence twice.

    Returns
    -------
    result : Any
        Object returned by the timed execution of ``func``.
    elapsed_seconds : float
        Wall-clock execution time measured with ``time.perf_counter``.
    peak_memory_mib : float
        Peak memory traced by ``tracemalloc`` during the memory pass, in MiB.
    """
    # -------------------------------------------------------------------------
    # Preserve RNG state so the second profiling pass follows the same random
    # sequence whenever the imputation implementation uses random / np.random.
    # -------------------------------------------------------------------------
    python_random_state_before = random.getstate()
    numpy_random_state_before = np.random.get_state()

    # -------------------------------------------------------------------------
    # Pass 1: execution time only (tracemalloc OFF).
    # -------------------------------------------------------------------------
    if tracemalloc.is_tracing():
        tracemalloc.stop()

    gc.collect()
    start_time = time.perf_counter()
    result = func()
    elapsed_seconds = time.perf_counter() - start_time

    python_random_state_after = random.getstate()
    numpy_random_state_after = np.random.get_state()

    # -------------------------------------------------------------------------
    # Pass 2: peak memory only (tracemalloc ON).
    # -------------------------------------------------------------------------
    random.setstate(python_random_state_before)
    np.random.set_state(numpy_random_state_before)
    gc.collect()

    tracemalloc.start()
    try:
        _ = func()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        # Profiling must not advance the experiment RNG state a second time.
        random.setstate(python_random_state_after)
        np.random.set_state(numpy_random_state_after)

    peak_memory_mib = peak_bytes / (1024 ** 2)

    return result, elapsed_seconds, peak_memory_mib

def _linear_series(s):
    """Linear interpolation with boundary fallback."""
    return (
        s.astype(float)
        .interpolate(method='linear', limit_direction='both')
        .ffill()
        .bfill()
    )


def _observed_xy(s):
    y = s.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    mask = np.isfinite(y)
    return x, y, mask

# =============================================================================
# Fixed benchmark imputation methods
# =============================================================================

def impute_linear(df):
    out = df.copy()
    for col in out.columns:
        out[col] = _linear_series(out[col])
    return out


def impute_pchip(df):
    out = df.copy()

    for col in out.columns:
        s = out[col].astype(float)
        x, y, observed = _observed_xy(s)

        if observed.sum() < 2:
            out[col] = _linear_series(s)
            continue

        try:
            interpolator = PchipInterpolator(
                x[observed],
                y[observed],
                extrapolate=True
            )
            missing = ~observed
            y_new = y.copy()
            y_new[missing] = interpolator(x[missing])
            out[col] = y_new
        except Exception:
            out[col] = _linear_series(s)

    return out


def impute_cubic_spline(df):
    out = df.copy()

    for col in out.columns:
        s = out[col].astype(float)
        x, y, observed = _observed_xy(s)

        if observed.sum() < 4:
            out[col] = _linear_series(s)
            continue

        try:
            interpolator = CubicSpline(
                x[observed],
                y[observed],
                bc_type='natural',
                extrapolate=True
            )
            missing = ~observed
            y_new = y.copy()
            y_new[missing] = interpolator(x[missing])
            out[col] = y_new
        except Exception:
            out[col] = _linear_series(s)

    return out


def impute_local_median(df, window=5):
    """
    Fill each missing point with the median of observed values in a local window.
    If the window contains no observed values, use linear interpolation as fallback.
    """
    out = df.copy()
    radius = max(1, window // 2)

    for col in out.columns:
        original = out[col].astype(float).copy()
        result = original.copy()
        fallback = _linear_series(original)
        missing_idx = np.flatnonzero(original.isna().to_numpy())

        for idx in missing_idx:
            left = max(0, idx - radius)
            right = min(len(original), idx + radius + 1)
            local_values = original.iloc[left:right].dropna()

            if len(local_values) > 0:
                result.iloc[idx] = float(local_values.median())
            else:
                result.iloc[idx] = fallback.iloc[idx]

        result = result.fillna(fallback)
        out[col] = result

    return out


def _estimate_seasonal_lag(s, max_lag=None):
    """
    Estimate dominant lag using autocorrelation of a temporarily linearly filled
    series. This is used only to choose a fixed seasonal-lag reconstruction.
    """
    filled = _linear_series(s).to_numpy(dtype=float)
    n = len(filled)

    if n < 5 or np.nanstd(filled) == 0:
        return 1

    if max_lag is None:
        max_lag = min(200, max(2, n // 3))
    max_lag = min(max_lag, n - 2)

    best_lag = 1
    best_corr = -np.inf

    for lag in range(2, max_lag + 1):
        a = filled[:-lag]
        b = filled[lag:]

        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            continue

        corr = np.corrcoef(a, b)[0, 1]
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag


def impute_seasonal_lag(df):
    """
    Fixed seasonal-lag baseline. For every missing point, use the available value
    one dominant lag before/after it; if both exist, average them. Remaining gaps
    fall back to linear interpolation.
    """
    out = df.copy()

    for col in out.columns:
        original = out[col].astype(float).copy()
        result = original.copy()
        fallback = _linear_series(original)
        lag = _estimate_seasonal_lag(original)
        missing_idx = np.flatnonzero(original.isna().to_numpy())

        for idx in missing_idx:
            candidates = []

            prev_idx = idx - lag
            next_idx = idx + lag

            if prev_idx >= 0 and pd.notna(original.iloc[prev_idx]):
                candidates.append(float(original.iloc[prev_idx]))

            if next_idx < len(original) and pd.notna(original.iloc[next_idx]):
                candidates.append(float(original.iloc[next_idx]))

            if candidates:
                result.iloc[idx] = float(np.mean(candidates))
            else:
                result.iloc[idx] = fallback.iloc[idx]

        result = result.fillna(fallback)
        out[col] = result

    return out


def impute_moving_average(df, window=5):
    """
    Moving-average baseline.

    Missing values are imputed using a centered rolling mean computed
    from the available observations in the local neighbourhood.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with missing values.

    window : int, default=5
        Size of the rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed.
    """
    out = df.copy()

    for col in out.columns:
        s = out[col].astype(float)

        # If there is not enough information, use linear interpolation.
        if s.notna().sum() < 2:
            out[col] = _linear_series(s)
            continue

        rolling_mean = s.rolling(
            window=window,
            center=True,
            min_periods=1
        ).mean()

        result = s.copy()
        missing = s.isna()

        # Fill only originally missing values.
        result.loc[missing] = rolling_mean.loc[missing]

        # For longer gaps the centered rolling mean can still return NaN
        # when the whole local window contains missing observations.
        # Use linear interpolation as a fallback.
        result = result.fillna(_linear_series(s))

        out[col] = result

    return out

# =============================================================================
# Evaluation ONLY at artificially removed positions
# =============================================================================
def _safe_mae(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def _safe_rmse(y_true, y_pred):
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compare_on_missing_only(original, incomplete, imputed, benchmark_method):
    """
    Compare original and imputed series ONLY at positions that were
    artificially removed in `incomplete`.

    Metrics:
    - MAE
    - RMSE
    """

    original = similarity.__ensure_dataframe(original)
    incomplete = similarity.__ensure_dataframe(incomplete)
    imputed = similarity.__ensure_dataframe(imputed)

    results = []

    for col in original.columns:
        mask = incomplete[col].isna() & original[col].notna()

        y_true = original.loc[mask, col].astype(float).to_numpy()
        y_pred = imputed.loc[mask, col].astype(float).to_numpy()

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        row = {
            "series": col,
            "benchmark_method": benchmark_method,
            "n_missing_evaluated": int(len(y_true)),
            "MAE": _safe_mae(y_true, y_pred),
            "RMSE": _safe_rmse(y_true, y_pred)
        }

        results.append(row)

    return pd.DataFrame(results)

# =============================================================================
# Compare original vs imputed
# =============================================================================
def compare_via_pair_df(original, imputed):

    original = similarity.__ensure_dataframe(original)
    imputed = similarity.__ensure_dataframe(imputed)

    results = []

    for col in original.columns:

        ts1 = original[col]
        ts2 = imputed[col]

        ts1, ts2 = ts1.align(ts2, join='inner')

        pair_df = pd.DataFrame({
            "original": ts1,
            "imputed": ts2
        })

        metrics_dict = similarity.similarity_metrics(pair_df)

        row = {"series": col}

        for metric_name, df in metrics_dict.items():
            row[metric_name] = df.loc["original", "imputed"]

        results.append(row)

    return pd.DataFrame(results)

# =============================================================================
# Delete functions
# =============================================================================
def delete_every_nth_per_column(time_series, percentage):

    result = time_series.copy()

    length_ts = len(time_series)

    params = []
    for col in result.columns:
    
        start_index = random.randint(1, 200)
        
        remaining = length_ts - start_index
        
        target_missing = int(remaining * percentage / 100)
        
        if target_missing == 0:
            continue
        
        max_n = max(2, remaining // target_missing)

        n_upper = min(max_n, 10)
        
        n = random.randint(2, n_upper)
        
        m = random.randint(0, n - 1)
        
        possible_missing = len(range(start_index + m, length_ts, n))
        
        if possible_missing < target_missing:
            continue
    
        try:
    
            col_df = delete.delete_every_nth(
                result[[col]],
                start_index,
                percentage,
                n,
                m,
                index=None
            )
    
            result[col] = col_df[col]
    
            params.append({
                "series": col,
                "start_index": start_index,
                "n": n,
                "m": m,
                "percentage": percentage,
                "method": "every_nth"
            })
    
        except ValueError as e:
    
            print(
                f"ERROR: col={col}, "
                f"percentage={percentage}, "
                f"n={n}, m={m}, error={e}"
            )

    return result, pd.DataFrame(params)


def delete_random_per_column(time_series, percentage):

    result = time_series.copy()

    length_ts = len(time_series)

    params = []

    for col in result.columns:

        start_index = random.randint(1, 200)

        end_index = random.randint(700, length_ts - 1)

        seed = random.randint(0, 10000)

        col_df = delete.delete_random(
            result[[col]],
            start_index,
            end_index,
            percentage,
            index=None,
            seed=seed
        )

        result[col] = col_df[col]

        params.append({
            "series": col,
            "start_index": start_index,
            "end_index": end_index,
            "percentage": percentage,
            "seed": seed,
            "method": "random"
        })

    return result, pd.DataFrame(params)


def delete_subsequence_per_column(time_series, percentage):

    result = time_series.copy()

    params = []

    for col in result.columns:

        start_index = random.randint(1, 200)

        col_df = delete.delete_subsequence(
            result[[col]],
            start_index,
            percentage,
            index=None
        )

        result[col] = col_df[col]

        params.append({
            "series": col,
            "start_index": start_index,
            "percentage": percentage,
            "method": "subsequence"
        })

    return result, pd.DataFrame(params)

# =============================================================================
# Percentage and missing methods executions
# =============================================================================
percentage = int(data['percentage'].iloc[0])

methods = {
    "every_nth": lambda ts: delete_every_nth_per_column(ts,percentage),
    "random": lambda ts: delete_random_per_column(ts,percentage),
    "subsequence": lambda ts: delete_subsequence_per_column(ts,percentage)
}


# =============================================================================
# Main loop
# =============================================================================
for name, func in methods.items():

    # =============================================================================
    # Create directories
    # =============================================================================
    input_subdir = os.path.join(input_dir, name)
    output_subdir = os.path.join(output_dir, name)

    os.makedirs(input_subdir, exist_ok=True)
    os.makedirs(output_subdir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Create ONE incomplete dataset shared by all benchmark methods
    # -------------------------------------------------------------------------
    time_series_del, params_df = func(time_series)

    time_series_del_path = os.path.join(input_subdir, 'time_series_del.csv')
    save_df(time_series_del, time_series_del_path)

    params_path = os.path.join(output_subdir, 'deletion_params.csv')
    params_df.to_csv(params_path, index=False)

    all_comparisons = []

    # One resource row per imputation method for this missingness mechanism.
    resource_metrics = defaultdict(list)
    
    
    # =============================================================================
    # Imputation - adaptive method
    # =============================================================================
    adaptive_result, adaptive_time_sec, adaptive_peak_memory_mib = profile_callable(
        lambda: impute.impute_missing_pipeline(time_series_del.copy())
    )
    adaptive_df, report_df = adaptive_result

    resource_metrics['benchmark_method'].append('Adaptive')
    resource_metrics['execution_time_sec'].append(adaptive_time_sec)
    resource_metrics['peak_memory_mib'].append(adaptive_peak_memory_mib)

    adaptive_comparison = compare_on_missing_only(
        time_series,
        time_series_del,
        adaptive_df,
        'Adaptive'
    )
    adaptive_comparison['execution_time_sec'] = adaptive_time_sec
    adaptive_comparison['peak_memory_mib'] = adaptive_peak_memory_mib
    all_comparisons.append(adaptive_comparison)

    save_df(
        report_df,
        os.path.join(output_subdir, 'report_imputation.csv')
    )

    # Keep legacy adaptive output filename for compatibility
    save_df(
        adaptive_df,
        os.path.join(output_subdir, 'time_series_impute.csv')
    )
    
    # =============================================================================
    # Compare - adaptive df (my tool)
    # =============================================================================
    comparison_df = compare_via_pair_df(
        time_series,
        adaptive_df
    )

    comparison_path = os.path.join(
        output_subdir,
        'comparison_metrics_similarity.csv'
    )

    comparison_df.to_csv(comparison_path, index=False)

    # -------------------------------------------------------------------------
    # 2-7. Fixed benchmark methods
    # -------------------------------------------------------------------------
    fixed_methods = {
        'Linear': impute_linear,
        'PCHIP': impute_pchip,
        'Cubic_spline': impute_cubic_spline,
        'Local_median': impute_local_median,
        'Seasonal_lag': impute_seasonal_lag,
        'Moving_average': impute_moving_average,
    }
    
    for benchmark_method, imputation_func in fixed_methods.items():  
        try:
            fixed_imputed_df, execution_time_sec, peak_memory_mib = profile_callable(
                lambda: imputation_func(time_series_del.copy())
            )

            resource_metrics['benchmark_method'].append(benchmark_method)
            resource_metrics['execution_time_sec'].append(execution_time_sec)
            resource_metrics['peak_memory_mib'].append(peak_memory_mib)

            comparison_df = compare_on_missing_only(
                time_series,
                time_series_del,
                fixed_imputed_df,
                benchmark_method
            )

            # Repeated on each series row for convenient downstream aggregation.
            comparison_df['execution_time_sec'] = execution_time_sec
            comparison_df['peak_memory_mib'] = peak_memory_mib

            all_comparisons.append(comparison_df)
    
        except Exception as exc:
            print(
                f"ERROR in benchmark method {benchmark_method} "
                f"for missingness {name}: {exc}"
            )

            resource_metrics['benchmark_method'].append(benchmark_method)
            resource_metrics['execution_time_sec'].append(np.nan)
            resource_metrics['peak_memory_mib'].append(np.nan)
    
            # Preserve a row for failed method so the benchmark is auditable.
            failure_rows = []
            for col in time_series.columns:
                n_missing = int(
                    (time_series_del[col].isna() & time_series[col].notna()).sum()
                )
                failure_rows.append({
                    'series': col,
                    'benchmark_method': benchmark_method,
                    'n_missing_evaluated': n_missing,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'execution_time_sec': np.nan,
                    'peak_memory_mib': np.nan,
                    'benchmark_error': str(exc),
                })
    
            all_comparisons.append(pd.DataFrame(failure_rows))

    # -------------------------------------------------------------------------
    # Save resource usage separately: exactly one row per benchmark method.
    # -------------------------------------------------------------------------
    resource_metrics_df = pd.DataFrame(resource_metrics)
    resource_metrics_df.insert(0, 'missingness_method', name)
    resource_metrics_df.to_csv(
        os.path.join(output_subdir, 'resource_metrics.csv'),
        index=False
    )

    # -------------------------------------------------------------------------
    # Save ONE comparison file containing all 7 benchmark methods.
    # Existing aggregation script can read this file; benchmark_method remains
    # separate from the experiment/configuration column named `method`.
    # -------------------------------------------------------------------------
    comparison_all_df = pd.concat(all_comparisons, ignore_index=True, sort=False)
    
    preferred_columns = [
        'series',
        'benchmark_method',
        'n_missing_evaluated',
        'MAE',
        'RMSE',
        'execution_time_sec',
        'peak_memory_mib'
    ]
    
    preferred_existing = [
        c for c in preferred_columns if c in comparison_all_df.columns
    ]
    other_columns = [
        c for c in comparison_all_df.columns if c not in preferred_existing
    ]
    comparison_all_df = comparison_all_df[preferred_existing + other_columns]
    
    comparison_path = os.path.join(output_subdir, 'comparison_metrics.csv')
    comparison_all_df.to_csv(comparison_path, index=False)
