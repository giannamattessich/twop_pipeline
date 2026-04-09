from twop_pipeline.lfp.lfpFunctions import *

def summarize_bands(
    spect_power_db,
    f,
    bands=(("delta", 0.5, 4),
           ("theta", 4, 8.5),
           ("alpha", 8.5, 13),
           ("beta", 15, 30),
           ("gamma", 30, 80),
           ("broadband", 0.5, 120)),
    normalize=True,):
    """
    Summarize band power from a spectrogram.

    Parameters
    ----------
    spect_power_db : array-like, shape (F, T) or (T, F)
        Spectral power in dB.
    f : array-like, shape (F,)
        Frequency vector (Hz).
    bands : tuple
        Iterable of (name, low, high).
    normalize : bool
        If True, also compute per-band z-scored time series and store as '{band}_z'.

    Returns
    -------
    out : dict
        Keys: '{band}' -> band mean power (dB) over time (T,)
              and if normalize: '{band}_z' -> z-scored version over time (T,)
    """
    Sdb, f = ensure_freq_first(spect_power_db, f)  # ensures (F, T)
    Sdb = np.asarray(Sdb, dtype=float)
    f = np.asarray(f, dtype=float)

    # Clean non-finite values (±inf -> NaN)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    F, T = Sdb.shape
    out = {}

    for name, low, high in bands:
        m = get_band(f, low, high)

        if not np.any(m):
            band_ts = np.full(T, np.nan)
        else:
            band_slice = Sdb[m, :]  # (F_band, T)
            band_ts = np.nanmean(band_slice, axis=0) if not np.all(np.isnan(band_slice)) else np.full(T, np.nan)

        out[name] = band_ts

        if normalize:
            # z-score *within this band time series* (across time)
            mu = np.nanmean(band_ts)
            sd = np.nanstd(band_ts) + 1e-12
            out[f"{name}_z"] = (band_ts - mu) / sd

    return out

def summarize_linear_power(
    spect_power_db,
    f,
    bands=(("delta", 0.5, 4),
           ("theta", 4, 8.5),
           ("alpha", 8.5, 13),
           ("beta", 15, 30),
           ("gamma", 30, 80),
           ("broadband", 0.5, 120)),
    normalize=True,
    eps=1e-12,
):
    """
    Summarize band power from a spectrogram using LINEAR power aggregation.

    Parameters
    ----------
    spect_power_db : array-like, shape (F, T) or (T, F)
        Spectral power in dB.
    f : array-like, shape (F,)
        Frequency vector (Hz).
    bands : tuple
        Iterable of (name, low, high).
    normalize : bool
        If True, also compute per-band z-scored time series and store as '{band}_z'.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    out : dict
        Keys:
          '{band}'    -> band power over time in dB (T,)
          '{band}_lin'-> band power over time in LINEAR units (T,)
          '{band}_z'  -> z-scored dB band power (T,) [if normalize=True]
    """
    # Ensure frequency is first axis → (F, T)
    Sdb, f = ensure_freq_first(spect_power_db, f)
    Sdb = np.asarray(Sdb, dtype=float)
    f = np.asarray(f, dtype=float)

    # Clean non-finite values
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    # Convert FULL spectrogram to linear power ONCE
    Slin = 10 ** (Sdb / 10.0)

    F, T = Slin.shape
    out = {}

    for name, low, high in bands:
        m = get_band(f, low, high)

        if not np.any(m):
            band_lin = np.full(T, np.nan)
        else:
            band_slice = Slin[m, :]  # (F_band, T)
            band_lin = np.nanmean(band_slice, axis=0)

        # Store linear power
        out[f"{name}_lin"] = band_lin

        # Convert back to dB (for plotting / z-score)
        band_db = 10 * np.log10(band_lin + eps)
        out[name] = band_db

        if normalize:
            mu = np.nanmean(band_db)
            sd = np.nanstd(band_db) + eps
            out[f"{name}_z"] = (band_db - mu) / sd
    return out


def overall_metrics(Sdb, f):
    """
    Return robust scalars for easy cross-day comparison.
    Accepts Sdb shaped (F, T) or (T, F); detects frequency axis.
    """
    Sdb, f = ensure_freq_first(Sdb, f)  # Sdb -> (F, T)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    def band_mean(low, high):
        m = get_band(f, low, high)
        if not np.any(m):
            return np.nan
        vals = Sdb[m, :]  # (F_band, T)
        if np.all(np.isnan(vals)):
            return np.nan
        return np.nanmean(vals)  # mean over freq + time

    bb   = band_mean(0.5, 120)   # broadband
    gam  = band_mean(30, 80)
    delt = band_mean(0.5, 4)
    thet = band_mean(4, 8.5)
    beta = band_mean(15,30)
    alpha = band_mean(8,12)

    # dB difference (≈ log10 gamma/delta ratio)
    gdr = gam - delt if np.isfinite(gam) and np.isfinite(delt) else np.nan

    # helper: convert dB → linear and compute ratio safely
    def ratio_from_db(num_db, den_db):
        if not (np.isfinite(num_db) and np.isfinite(den_db)):
            return np.nan
        num_lin = 10.0 ** (num_db / 10.0)
        den_lin = 10.0 ** (den_db / 10.0)
        if den_lin == 0:
            return np.nan
        return num_lin / den_lin

    delta_theta_ratio = ratio_from_db(delt, thet)
    delta_beta_ratio  = ratio_from_db(delt, beta)
    theta_gamma_ratio = ratio_from_db(thet, gam)

    return dict(
        broadband_db=bb,
        gamma_db=gam,
        delta_db=delt,
        theta_db=thet,
        beta_db=beta,
        alpha_db=alpha,
        gamma_delta_diff_db=gdr,
        delta_theta_ratio=delta_theta_ratio,
        delta_beta_ratio=delta_beta_ratio,
        theta_gamma_ratio=theta_gamma_ratio,)


def lfp_feature_frame(Sdb, f, t):
    Sdb, f = ensure_freq_first(Sdb, f)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    def band_ts(low, high):
        m = get_band(f, low, high)
        if not np.any(m):
            return np.full(Sdb.shape[1], np.nan)
        vals = Sdb[m, :]
        return np.nanmean(vals, axis=0)

    bb    = band_ts(0.5, 120)
    gam   = band_ts(30, 80)
    delt  = band_ts(0.5, 4)
    thet  = band_ts(4, 8.5)
    beta  = band_ts(15, 30)
    alpha = band_ts(8, 12)

    gamma_delta_diff_db = gam - delt

    delta_lin = db_to_linear_power(delt)
    theta_lin = db_to_linear_power(thet)
    beta_lin  = db_to_linear_power(beta)
    gamma_lin = db_to_linear_power(gam)

    return pd.DataFrame({
        "time": t,
        "broadband_db": bb,
        "gamma_db": gam,
        "delta_db": delt,
        "theta_db": thet,
        "beta_db": beta,
        "alpha_db": alpha,
        "gamma_delta_diff_db": gamma_delta_diff_db,
        "delta_theta_ratio": compute_linear_power_ratio(delta_lin, theta_lin),
        "theta_delta_ratio": compute_linear_power_ratio(theta_lin, delta_lin),
        "delta_beta_ratio": compute_linear_power_ratio(delta_lin, beta_lin),
        "theta_gamma_ratio": compute_linear_power_ratio(theta_lin, gamma_lin),
    })


def build_lfp_feature_df(
    Sdb,
    f,
    t,
    bands=None,
    include_linear_bandpowers=False,
):
    """
    Build a time-resolved LFP feature dataframe from a spectrogram.

    Parameters
    ----------
    Sdb : np.ndarray
        Spectrogram in dB, shape (freq, time) or (time, freq)
    f : np.ndarray
        Frequency vector
    t : np.ndarray
        Spectrogram time vector, one value per time bin
    bands : dict or None
        Mapping like {"delta": (0.5, 4), ...}
    include_linear_bandpowers : bool
        If True, also include *_lin columns

    Returns
    -------
    lfp_df : pd.DataFrame
        One row per spectrogram time bin
    """
    Sdb, f = ensure_freq_first(Sdb, f)
    t = np.asarray(t)

    n_time = Sdb.shape[1]
    if len(t) != n_time:
        raise ValueError(f"len(t)={len(t)} does not match spectrogram n_time={n_time}")

    if bands is None:
        bands = {
            "broadband": (0.5, 120),
            "gamma": (30, 80),
            "delta": (0.5, 4),
            "theta": (4, 8.5),
            "beta": (15, 30),
            "alpha": (8, 12),
        }

    out = {"time": t}
    band_db = {}
    band_lin = {}

    for band_name, (low, high) in bands.items():
        band_db[band_name] = band_mean_db_over_time(Sdb, f, low, high)
        band_lin[band_name] = band_mean_linear_over_time(Sdb, f, low, high)

        out[f"{band_name}_db"] = band_db[band_name]
        if include_linear_bandpowers:
            out[f"{band_name}_lin"] = band_lin[band_name]

    if "gamma" in band_db and "delta" in band_db:
        out["gamma_delta_diff_db"] = band_db["gamma"] - band_db["delta"]

    if "delta" in band_lin and "theta" in band_lin:
        out["delta_theta_ratio"] = safe_ratio(band_lin["delta"], band_lin["theta"])
        out["theta_delta_ratio"] = safe_ratio(band_lin["theta"], band_lin["delta"])

    if "delta" in band_lin and "beta" in band_lin:
        out["delta_beta_ratio"] = safe_ratio(band_lin["delta"], band_lin["beta"])

    if "theta" in band_lin and "gamma" in band_lin:
        out["theta_gamma_ratio"] = safe_ratio(band_lin["theta"], band_lin["gamma"])

    return pd.DataFrame(out)


def resample_lfp_df_to_times(
    lfp_df,
    target_times,
    time_col="time",
    method="interp",
):
    """
    Resample / interpolate a time-resolved LFP dataframe to a target timebase.

    Parameters
    ----------
    lfp_df : pd.DataFrame
        Output of build_lfp_feature_df()
    target_times : array-like
        Times you want features at, e.g. motion times or scope/frame times
    time_col : str
        Name of time column in lfp_df
    method : str
        Currently supports:
        - "interp": linear interpolation onto target_times
        - "nearest": nearest-neighbor assignment

    Returns
    -------
    out_df : pd.DataFrame
        DataFrame with one row per target time
    """
    target_times = np.asarray(target_times, dtype=float)
    src_t = np.asarray(lfp_df[time_col].values, dtype=float)

    if np.any(np.diff(src_t) < 0):
        order = np.argsort(src_t)
        lfp_df = lfp_df.iloc[order].reset_index(drop=True)
        src_t = np.asarray(lfp_df[time_col].values, dtype=float)

    out = {time_col: target_times}

    for col in lfp_df.columns:
        if col == time_col:
            continue

        y = np.asarray(lfp_df[col].values, dtype=float)
        valid = np.isfinite(src_t) & np.isfinite(y)

        if np.sum(valid) < 2:
            out[col] = np.full(len(target_times), np.nan)
            continue

        src_t_valid = src_t[valid]
        y_valid = y[valid]

        if method == "interp":
            out_arr = np.interp(target_times, src_t_valid, y_valid)
            outside = (target_times < src_t_valid[0]) | (target_times > src_t_valid[-1])
            out_arr[outside] = np.nan
            out[col] = out_arr

        elif method == "nearest":
            idx = np.searchsorted(src_t_valid, target_times)

            idx0 = np.clip(idx - 1, 0, len(src_t_valid) - 1)
            idx1 = np.clip(idx,     0, len(src_t_valid) - 1)

            d0 = np.abs(target_times - src_t_valid[idx0])
            d1 = np.abs(target_times - src_t_valid[idx1])

            choose1 = d1 < d0
            nearest_idx = idx0.copy()
            nearest_idx[choose1] = idx1[choose1]

            out_arr = y_valid[nearest_idx]
            outside = (target_times < src_t_valid[0]) | (target_times > src_t_valid[-1])
            out_arr[outside] = np.nan
            out[col] = out_arr

        else:
            raise ValueError("method must be 'interp' or 'nearest'")

    return pd.DataFrame(out)


def make_lfp_features(
    Sdb,
    f,
    lfp_times,
    target_times=None,
    target_name="time",
    bands=None,
    include_linear_bandpowers=False,
    resample_method="interp",
):
    """
    Build LFP features and optionally resample to a target timebase.

    Parameters
    ----------
    Sdb : np.ndarray
        Spectrogram in dB
    f : np.ndarray
        Frequency vector
    lfp_times : np.ndarray
        Time vector for the spectrogram bins
    target_times : np.ndarray or None
        Optional target times, e.g. motion/camera or 2P scope times
    target_name : str
        Name of output time column if resampling
    bands : dict or None
        Frequency bands
    include_linear_bandpowers : bool
        Whether to keep *_lin columns
    resample_method : str
        'interp' or 'nearest'

    Returns
    -------
    lfp_df : pd.DataFrame
        If target_times is None, returns spectrogram-time dataframe.
        Otherwise returns dataframe resampled onto target_times.
    """
    lfp_df = build_lfp_feature_df(
        Sdb=Sdb,
        f=f,
        t=lfp_times,
        bands=bands,
        include_linear_bandpowers=include_linear_bandpowers,
    )

    if target_times is None:
        return lfp_df

    out_df = resample_lfp_df_to_times(
        lfp_df=lfp_df,
        target_times=target_times,
        time_col="time",
        method=resample_method,
    )

    if target_name != "time":
        out_df = out_df.rename(columns={"time": target_name})

    return out_df