import os, numpy as np, traceback
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from twop_pipeline.lfp.readLFP import *

def channel_reduce(arr, method="median"):
    """
    Reduce channels to a single 1D time-series.
    Accepts (T,), (T,C) or (C,T) and returns (T,).

    method: 'median' or 'mean'
    """
    x = np.asarray(arr)
    if method == "median":
        return np.median(x, axis=1)  # (T,)
    elif method == "mean":
        return np.mean(x, axis=1)    # (T,)
    else:
        raise ValueError("method must be 'median' or 'mean'")

def get_band(freq, low, high):
    freq = np.asarray(freq)
    return (freq >= low) & (freq <= high)

def safe_ratio(num, den, eps=1e-12):
    """Safe elementwise ratio."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[valid] = num[valid] / den[valid]
    return out

def ensure_freq_first(Sdb, f):
    """
    Ensure Sdb is shaped (F, T), where F == len(f).
    If Sdb is (T, F), it will be transposed automatically.
    """
    Sdb = np.asarray(Sdb, dtype=float)
    f = np.asarray(f, dtype=float)

    if Sdb.ndim != 2:
        raise ValueError(f"Sdb must be 2D (F, T or T, F), got shape {Sdb.shape}")

    F = f.size
    if Sdb.shape[0] == F:
        # Already (F, T)
        return Sdb, f
    elif Sdb.shape[1] == F:
        # Likely (T, F) -> transpose
        return Sdb.T, f
    else:
        raise ValueError(
            f"None of Sdb's axes match len(f). Sdb.shape={Sdb.shape}, len(f)={F}")


def normalize_per_freq(spect_power_db):
    """Z-score per frequency row (so color maps are comparable across days)."""
    Sdb = np.asarray(spect_power_db, dtype=float)

    # Replace ±inf with NaN so they don't break mean/std
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    mu = np.nanmean(Sdb, axis=1, keepdims=True)
    sd = np.nanstd(Sdb, axis=1, keepdims=True) + 1e-12
    return (Sdb - mu) / sd

def linear_power_to_db(x_lin, eps=1e-12):
    """Convert linear power to dB safely."""
    x_lin = np.asarray(x_lin, dtype=float)
    return 10.0 * np.log10(np.maximum(x_lin, eps))

def band_mean_db_over_time(Sdb, f, low, high):
    """
    Mean band power in dB for each spectrogram time bin.

    Returns
    -------
    band_db : np.ndarray, shape (n_time,)
    """
    Sdb, f = ensure_freq_first(Sdb, f)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    mask = get_band(f, low, high)
    if not np.any(mask):
        return np.full(Sdb.shape[1], np.nan)

    return np.nanmean(Sdb[mask, :], axis=0)

def band_mean_linear_over_time(Sdb, f, low, high):
    """
    Mean band power in linear space for each spectrogram time bin.

    This is preferable for computing ratios.
    """
    Sdb, f = ensure_freq_first(Sdb, f)
    Slin = db_to_linear_power(np.where(np.isfinite(Sdb), Sdb, np.nan))

    mask = get_band(f, low, high)
    if not np.any(mask):
        return np.full(Sdb.shape[1], np.nan)

    return np.nanmean(Slin[mask, :], axis=0)

def compute_spectrogram(
    lfp,
    fs,
    channel=0,
    nperseg=2048,
    noverlap=1536,
    fmax=200,
    scaling="spectrum",
    window="hann",
    detrend=False,
):
    """
    Compute LFP power spectrogram in dB.

    Parameters
    ----------
    lfp : array-like
        1D (n_samples,) or 2D ((n_samples, n_channels) or (n_channels, n_samples))
    fs : float
        Sampling rate in Hz
    channel : int
        Which channel to use if lfp is 2D
    nperseg : int
        Samples per FFT window
    noverlap : int
        Overlap in samples
    fmax : float or None
        Max frequency to keep
    scaling : str
        'density' or 'spectrum'
    window : str or array-like
        Window passed to scipy.signal.spectrogram
    detrend : bool or str
        Detrending passed to scipy.signal.spectrogram

    Returns
    -------
    freqs : np.ndarray
        Frequencies (Hz)
    times : np.ndarray
        Spectrogram times (s)
    power_db : np.ndarray
        Power spectrogram in dB, shape (freqs, times)
    """
    lfp = np.asarray(lfp)

    # ensure 1D signal
    if lfp.ndim == 2:
        if lfp.shape[0] > lfp.shape[1]:
            sig = lfp[:, channel]   # (samples, channels)
        else:
            sig = lfp[channel, :]   # (channels, samples)
    elif lfp.ndim == 1:
        sig = lfp
    else:
        raise ValueError(f"lfp must be 1D or 2D, got shape {lfp.shape}")

    sig = np.asarray(sig).squeeze()

    if nperseg > len(sig):
        nperseg = len(sig)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    freqs, times, power = spectrogram(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        detrend=detrend,
        window=window,
        mode="psd",   # explicit power spectral density
    )

    power_db = 10 * np.log10(np.maximum(power, 1e-20))

    if fmax is not None:
        mask = freqs <= fmax
        freqs = freqs[mask]
        power_db = power_db[mask, :]
    return freqs, times, power_db


def db_to_linear_power(power_db):
    """
    Convert power from decibels (dB) to linear scale.

    Parameters
    ----------
    power_db : array-like or float
        Power in dB.

    Returns
    -------
    power_linear : array-like or float
        Power in linear units.
    """
    return 10.0 ** (power_db / 10.0)


def compute_linear_power_ratio(numerator_db, denominator_db, eps=1e-12):
    """
    Compute a power ratio in linear space given numerator and denominator in dB.

    Parameters
    ----------
    numerator_db : array-like
        Power in dB for the numerator band.
    denominator_db : array-like
        Power in dB for the denominator band.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    ratio_linear : array-like
        Linear power ratio (numerator / denominator).
    """
    numerator_linear = db_to_linear_power(numerator_db)
    denominator_linear = db_to_linear_power(denominator_db)

    return numerator_linear / (denominator_linear + eps)


# def enforce_min_state_duration(state_sequence, minimum_duration=3):
#     """
#     Enforce a minimum duration for contiguous state segments.

#     Short state bouts (< minimum_duration) are replaced with the
#     preceding state if available, otherwise the following state.

#     Parameters
#     ----------
#     state_sequence : array-like of int
#         Discrete state labels over time.
#     minimum_duration : int
#         Minimum number of consecutive time bins required for a state.

#     Returns
#     -------
#     cleaned_states : ndarray
#         State sequence with short bouts removed.
#     """
#     cleaned_states = state_sequence.copy()
#     num_timepoints = len(cleaned_states)

#     start_idx = 0
#     while start_idx < num_timepoints:
#         end_idx = start_idx + 1

#         # Find end of current contiguous state block
#         while end_idx < num_timepoints and cleaned_states[end_idx] == cleaned_states[start_idx]:
#             end_idx += 1

#         bout_length = end_idx - start_idx

#         # Replace short bouts
#         if bout_length < minimum_duration:
#             if start_idx > 0:
#                 replacement_state = cleaned_states[start_idx - 1]
#             elif end_idx < num_timepoints:
#                 replacement_state = cleaned_states[end_idx]
#             else:
#                 replacement_state = cleaned_states[start_idx]

#             cleaned_states[start_idx:end_idx] = replacement_state

#         start_idx = end_idx

#     return cleaned_states

def resample_to_LFP(signal, signal_fs, n_lfp_samples, lfp_fs=1250):
    t_signal = np.arange(len(signal)) / signal_fs
    t_lfp    = np.arange(n_lfp_samples) / lfp_fs

    # interpolate signal onto LFP time grid
    signal_resampled = np.interp(t_lfp, t_signal, signal)

    return t_lfp, signal_resampled

def plv(phi):
    return np.abs(np.mean(np.exp(1j*phi)))

def rayleigh_p(phi):
    n = len(phi)
    R = n * plv(phi)
    # Small-sample corrected approximation
    z = (R**2)/n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    return max(min(p,1.0), 0.0)

# def compute_spectrogram(lfp, fs, win_sec=4, overlap_sec=2, channel=0):
#     """
#     lfp: 1D (n_samples,) or 2D (n_samples, n_channels) or (n_channels, n_samples)
#     fs : sampling rate

#     Output:
#         t (times)
#         f (freqs)
#         spec (sdb)
#     """

#     lfp = np.asarray(lfp)

#     # --- ensure 1D signal ---
#     if lfp.ndim == 2:
#         # Decide which axis is samples vs channels
#         # Heuristic: more samples than channels → samples axis is 0
#         if lfp.shape[0] > lfp.shape[1]:
#             # shape (n_samples, n_channels)
#             sig = lfp[:, channel]
#         else:
#             # shape (n_channels, n_samples)
#             sig = lfp[channel, :]
#     elif lfp.ndim == 1:
#         sig = lfp
#     else:
#         raise ValueError(f"lfp must be 1D or 2D, got shape {lfp.shape}")

#     # --- window & overlap ---
#     nperseg = int(win_sec * fs)
#     if nperseg > len(sig):
#         nperseg = len(sig)

#     noverlap = int(overlap_sec * fs)
#     if noverlap >= nperseg:
#         noverlap = nperseg - 1

#     # --- spectrogram ---
#     f, t, Sxx = spectrogram(
#         sig,
#         fs=fs,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         nfft=None,         # let scipy choose
#         scaling="density",
#         mode="magnitude",
#     )

#     spec = Sxx.T  # time x freq
#     return t, f, spec
