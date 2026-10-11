"""
FallGuard AI - Sensor Feature Engineering Module
Calculates comprehensive time-domain, dynamic, and frequency-domain features
from 3-axis accelerometer and 3-axis gyroscope sliding windows.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from typing import List, Dict, Any, Union

# Feature column names list (will be populated deterministically)
FEATURE_COLUMNS: List[str] = []

def compute_fft_features(signal: np.ndarray, sampling_rate: float = 50.0) -> Dict[str, float]:
    """Compute dominant frequency, spectral energy, and spectral entropy using FFT."""
    n = len(signal)
    if n < 4:
        return {"dominant_freq": 0.0, "spectral_energy": 0.0, "spectral_entropy": 0.0}
    
    # Detrend signal
    sig_detrend = signal - np.mean(signal)
    fft_vals = np.abs(rfft(sig_detrend))
    fft_freqs = rfftfreq(n, d=1.0 / sampling_rate)
    
    # Exclude DC component (freq = 0)
    if len(fft_vals) > 1:
        power = fft_vals[1:] ** 2
        freqs = fft_freqs[1:]
    else:
        power = fft_vals ** 2
        freqs = fft_freqs
        
    spectral_energy = float(np.sum(power) / (n + 1e-9))
    
    # Dominant frequency
    if len(power) > 0 and np.sum(power) > 1e-9:
        dominant_freq = float(freqs[np.argmax(power)])
        
        # Spectral entropy
        prob = power / np.sum(power)
        prob = prob[prob > 0]
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)) / np.log2(len(prob) + 1e-12))
    else:
        dominant_freq = 0.0
        spectral_entropy = 0.0
        
    return {
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "spectral_entropy": spectral_entropy
    }

def extract_channel_features(signal: np.ndarray, prefix: str, sampling_rate: float = 50.0) -> Dict[str, float]:
    """Extract full statistical and frequency features for a single 1D sensor channel."""
    features = {}
    n = len(signal)
    
    if n == 0:
        return features
        
    # Statistical measures
    mean_val = float(np.mean(signal))
    std_val = float(np.std(signal, ddof=1)) if n > 1 else 0.0
    var_val = float(np.var(signal, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    median_val = float(np.median(signal))
    range_val = float(max_val - min_val)
    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    energy_val = float(np.mean(signal ** 2))
    
    # Skewness & Kurtosis
    if std_val > 1e-7:
        skew_val = float(stats.skew(signal, bias=False))
        kurt_val = float(stats.kurtosis(signal, bias=False))
    else:
        skew_val = 0.0
        kurt_val = 0.0
        
    # Peak to peak
    p2p_val = range_val
    
    features[f"{prefix}_mean"] = mean_val
    features[f"{prefix}_std"] = std_val
    features[f"{prefix}_var"] = var_val
    features[f"{prefix}_min"] = min_val
    features[f"{prefix}_max"] = max_val
    features[f"{prefix}_median"] = median_val
    features[f"{prefix}_range"] = range_val
    features[f"{prefix}_rms"] = rms_val
    features[f"{prefix}_energy"] = energy_val
    features[f"{prefix}_p2p"] = p2p_val
    features[f"{prefix}_skew"] = skew_val
    features[f"{prefix}_kurtosis"] = kurt_val
    
    # FFT Frequency features
    fft_feat = compute_fft_features(signal, sampling_rate)
    features[f"{prefix}_dom_freq"] = fft_feat["dominant_freq"]
    features[f"{prefix}_spec_energy"] = fft_feat["spectral_energy"]
    features[f"{prefix}_spec_entropy"] = fft_feat["spectral_entropy"]
    
    return features

def extract_features_from_window(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    gyro_x: np.ndarray,
    gyro_y: np.ndarray,
    gyro_z: np.ndarray,
    sampling_rate: float = 50.0
) -> Dict[str, float]:
    """
    Extract comprehensive 6-axis wearable sensor features from a single window.
    
    Args:
        acc_x, acc_y, acc_z: 1D numpy arrays of accelerometer readings (in g or m/s²)
        gyro_x, gyro_y, gyro_z: 1D numpy arrays of gyroscope readings (in rad/s or deg/s)
        sampling_rate: sampling frequency in Hz
        
    Returns:
        Dictionary of feature_name: feature_value
    """
    features = {}
    
    # Magnitudes
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    # Channel feature extraction
    channels = [
        ("acc_x", acc_x),
        ("acc_y", acc_y),
        ("acc_z", acc_z),
        ("acc_mag", acc_mag),
        ("gyro_x", gyro_x),
        ("gyro_y", gyro_y),
        ("gyro_z", gyro_z),
        ("gyro_mag", gyro_mag)
    ]
    
    for name, data in channels:
        ch_feats = extract_channel_features(data, name, sampling_rate)
        features.update(ch_feats)
        
    # Signal Magnitude Area (SMA) and Signal Vector Magnitude (SVMag)
    n = len(acc_x)
    features["acc_sma"] = float(np.sum(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)) / n)
    features["gyro_sma"] = float(np.sum(np.abs(gyro_x) + np.abs(gyro_y) + np.abs(gyro_z)) / n)
    features["sma"] = features["acc_sma"]
    features["svmag_mean"] = float(np.mean(acc_mag))
    features["svmag_std"] = float(np.std(acc_mag, ddof=1)) if n > 1 else 0.0
    features["svmag_max"] = float(np.max(acc_mag)) if n > 0 else 0.0
    
    # Dynamic Features: Jerk (rate of change of acceleration)
    dt = 1.0 / sampling_rate
    jerk_x = np.diff(acc_x) / dt if n > 1 else np.zeros(1)
    jerk_y = np.diff(acc_y) / dt if n > 1 else np.zeros(1)
    jerk_z = np.diff(acc_z) / dt if n > 1 else np.zeros(1)
    jerk_mag = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
    
    features["jerk_mean"] = float(np.mean(jerk_mag))
    features["jerk_std"] = float(np.std(jerk_mag, ddof=1)) if len(jerk_mag) > 1 else 0.0
    features["jerk_max"] = float(np.max(jerk_mag)) if len(jerk_mag) > 0 else 0.0
    features["jerk_energy"] = float(np.mean(jerk_mag**2)) if len(jerk_mag) > 0 else 0.0
    
    # Angular Acceleration (rate of change of angular velocity)
    ang_acc_x = np.diff(gyro_x) / dt if n > 1 else np.zeros(1)
    ang_acc_y = np.diff(gyro_y) / dt if n > 1 else np.zeros(1)
    ang_acc_z = np.diff(gyro_z) / dt if n > 1 else np.zeros(1)
    ang_acc_mag = np.sqrt(ang_acc_x**2 + ang_acc_y**2 + ang_acc_z**2)
    
    features["ang_acc_mean"] = float(np.mean(ang_acc_mag))
    features["ang_acc_max"] = float(np.max(ang_acc_mag)) if len(ang_acc_mag) > 0 else 0.0
    
    # Tilt Angle Estimation (pitch and roll angles relative to vertical)
    pitch = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2)) * (180.0 / np.pi)
    roll = np.arctan2(-acc_x, acc_z) * (180.0 / np.pi)
    
    features["pitch_mean"] = float(np.mean(pitch))
    features["pitch_std"] = float(np.std(pitch, ddof=1)) if len(pitch) > 1 else 0.0
    features["pitch_range"] = float(np.ptp(pitch)) if len(pitch) > 0 else 0.0
    
    features["roll_mean"] = float(np.mean(roll))
    features["roll_std"] = float(np.std(roll, ddof=1)) if len(roll) > 1 else 0.0
    features["roll_range"] = float(np.ptp(roll)) if len(roll) > 0 else 0.0
    
    # Fall-specific impact indicators
    features["impact_ratio"] = float(features["acc_mag_max"] / (features["acc_mag_mean"] + 1e-5))
    features["post_impact_immobility"] = float(np.std(acc_mag[-int(n*0.3):])) if n >= 10 else 0.0
    
    return features

def extract_features_from_array(window_data: np.ndarray, sampling_rate: float = 50.0) -> Dict[str, float]:
    """
    Extract features from 2D numpy array of shape (N, 6)
    where columns are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z].
    """
    if window_data.shape[1] < 6:
        raise ValueError(f"Expected array with at least 6 columns, got shape {window_data.shape}")
        
    return extract_features_from_window(
        acc_x=window_data[:, 0],
        acc_y=window_data[:, 1],
        acc_z=window_data[:, 2],
        gyro_x=window_data[:, 3],
        gyro_y=window_data[:, 4],
        gyro_z=window_data[:, 5],
        sampling_rate=sampling_rate
    )

def extract_dataset_features(df_sensor_stream: pd.DataFrame, sampling_rate: float = 50.0) -> pd.DataFrame:
    """
    Extract windowed features for a full sensor stream DataFrame containing window_id.
    """
    grouped = df_sensor_stream.groupby("window_id")
    feature_rows = []
    
    for window_id, group in grouped:
        group_sorted = group.sort_values("step")
        
        ax = group_sorted["acc_x"].to_numpy()
        ay = group_sorted["acc_y"].to_numpy()
        az = group_sorted["acc_z"].to_numpy()
        gx = group_sorted["gyro_x"].to_numpy()
        gy = group_sorted["gyro_y"].to_numpy()
        gz = group_sorted["gyro_z"].to_numpy()
        
        feats = extract_features_from_window(ax, ay, az, gx, gy, gz, sampling_rate)
        
        # Meta info
        feats["window_id"] = window_id
        feats["subject_id"] = group_sorted["subject_id"].iloc[0]
        feats["activity"] = group_sorted["activity"].iloc[0]
        feats["is_fall"] = int(group_sorted["is_fall"].iloc[0])
        feats["split_origin"] = group_sorted["split_origin"].iloc[0] if "split_origin" in group_sorted else "train"
        
        feature_rows.append(feats)
        
    df_features = pd.DataFrame(feature_rows)
    return df_features
