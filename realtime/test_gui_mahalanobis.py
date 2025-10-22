import time
import threading
import pickle
import numpy as np
import mne
import logging
from pathlib import Path
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
from scipy.spatial.distance import mahalanobis
from scipy.stats import skew, kurtosis
from scipy.signal import welch, spectrogram
from antropy.entropy import spectral_entropy
import ordpy
from librosa.feature import mfcc
from PyEMD import EMD
import tkinter as tk
from tkinter import ttk
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------
# Configuration and Global Settings
# ---------------------------------------------------------------------
config = {
    "fs": 500,
    "l_freq": 1.0,
    "h_freq": 40.0,
    "n_fft": 512,
    "n_mfcc": 13,
    "window_size_sec": 2,
    "overlap": 0.5,
    "var_threshold": 1e-10
}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FS = config["fs"]
WINDOW_SIZE_SEC = config["window_size_sec"]
OVERLAP = config["overlap"]
WINDOW_SAMPLES = int(WINDOW_SIZE_SEC * FS)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))

# ---------------------------------------------------------------------
# Paths and Offline Model
# ---------------------------------------------------------------------
vhdr_file = r"G:\Raw Data\Final_new_data\19_Subject_Aditya\Run_2\Preprocessed\19_S__Baseline_raw.vhdr"
chunk_size = 10

kept_indices = np.load("kept_indices.npy") if Path("kept_indices.npy").exists() else None

with open("offline_model.pkl", "rb") as f:
    offline_model = pickle.load(f)
templates, cov_dict = offline_model["templates"], offline_model["cov_dict"]

with open("power_transformer.pkl", "rb") as f:
    transformer = pickle.load(f)

# ---------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------
global_info = None
global_buffer = None
buffer_lock = threading.Lock()

predictions_counter = {}
epochs_processed = 0

processing_active = threading.Event()
gui_queue = queue.Queue()

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
def preprocess_raw_data(raw, l_freq, h_freq, target_fs):
    """Preprocess raw EEG: resample, montage, filtering, interpolation."""
    if raw.info['sfreq'] > target_fs + 1:
        raw.resample(target_fs, npad="auto")

    try:
        raw.set_montage('easycap-M1', verbose=False)
    except Exception:
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), verbose=False)

    iir_params = dict(order=4, ftype='butter', output='sos')
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    raw.notch_filter(freqs=50, method='iir', iir_params=iir_params, phase='zero', verbose=False)
    raw.interpolate_bads(reset_bads=True, verbose=False)
    eeg_channels = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks=eeg_channels)
    return raw

# ---------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------
def compute_features(data, fs, eps=1e-6):
    """Extract features: statistics, spectral, entropy, complexity."""
    features = {
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'zero_crossing_rate': ((data[:-1] * data[1:]) < 0).sum()
    }

    diff1, diff2 = np.diff(data), np.diff(np.diff(data))
    activity = np.var(data)
    mobility = np.sqrt(np.var(diff1) / (activity + eps))
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + eps)) / (mobility + eps)
    features.update({
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    })

    freqs, psd = welch(data, fs=fs, nperseg=min(len(data), fs*2))
    total_power = np.sum(psd) + eps
    delta_power = np.sum(psd[(freqs >= 0.5) & (freqs <= 4)])
    beta_power = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
    theta_power = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
    alpha_power = np.sum(psd[(freqs >= 8) & (freqs <= 13)])

    features.update({
        'delta_relative_power': delta_power / total_power,
        'beta_relative_power': beta_power / total_power,
        'theta_alpha_ratio': theta_power / (alpha_power + eps),
        'beta_theta_ratio': beta_power / (theta_power + eps),
        'spectral_entropy': spectral_entropy(data, sf=fs, method='welch', normalize=True)
    })

    try:
        features['permutation_entropy'] = ordpy.permutation_entropy(data, dx=3)
    except Exception:
        features['permutation_entropy'] = np.nan

    def hurst(signal):
        N = len(signal)
        Y = np.cumsum(signal - np.mean(signal))
        R, S = np.max(Y) - np.min(Y), np.std(signal)
        return np.log(R / (S + eps) + eps) / np.log(N)

    def katz_fd(signal):
        L = np.sum(np.sqrt(1 + np.diff(signal) ** 2))
        d = np.max(np.abs(signal - signal[0]))
        N = len(signal)
        return np.log10(N) / (np.log10(N) + np.log10(d / (L + eps)))

    def higuchi_fd(signal, kmax=10):
        L, N = [], len(signal)
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = np.sum(np.abs(np.diff(signal[m::k]))) * (N - 1) / (len(signal[m::k]) * k)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        lnL, lnk = np.log(np.array(L) + eps), np.log(1.0 / np.arange(1, kmax + 1))
        return np.polyfit(lnk, lnL, 1)[0]

    features.update({
        'hurst_exponent': hurst(data),
        'katz_fd': katz_fd(data),
        'higuchi_fd': higuchi_fd(data)
    })

    mfccs = mfcc(y=data, sr=fs, n_mfcc=config["n_mfcc"], n_fft=config["n_fft"])
    for i in range(config["n_mfcc"]):
        features[f'mfcc_{i+1}'] = np.mean(mfccs[i, :])

    emd = EMD()
    imfs = emd(data)
    features['imf_1_entropy'] = spectral_entropy(imfs[0], sf=fs, method='welch', normalize=True) if len(imfs) > 0 else np.nan
    return features

def extract_all_features(epoch_data, fs):
    feats = compute_features(epoch_data, fs)
    return np.array(list(feats.values()))

# ---------------------------------------------------------------------
# Prediction (Mahalanobis)
# ---------------------------------------------------------------------
def predict_mahalanobis(sample, templates, cov_dict):
    """Return subject with minimum Mahalanobis distance."""
    best_subj, best_dist = None, float('inf')
    for subj, tmpl in templates.items():
        _, inv_cov = cov_dict[subj]
        dist = mahalanobis(sample, tmpl, inv_cov)
        if dist < best_dist:
            best_dist, best_subj = dist, subj
    return best_subj

# ---------------------------------------------------------------------
# Epoch Processing
# ---------------------------------------------------------------------
def process_epoch(epoch_buffer):
    global predictions_counter, epochs_processed, kept_indices
    features_matrix = [extract_all_features(epoch_buffer[:, ch], FS) for ch in range(epoch_buffer.shape[1])]
    sample_flat = np.vstack(features_matrix).flatten()
    sample_flat = np.nan_to_num(sample_flat, nan=0.0)
    if kept_indices is not None:
        sample_flat = sample_flat[kept_indices]
    transformed = transformer.transform(sample_flat.reshape(1, -1)).flatten()
    pred_subj = predict_mahalanobis(transformed, templates, cov_dict)

    predictions_counter[pred_subj] = predictions_counter.get(pred_subj, 0) + 1
    epochs_processed += 1
    gui_queue.put(("prediction", pred_subj))
    gui_queue.put(("log", f"Epoch {epochs_processed} processed. Prediction: {pred_subj}"))
    gui_queue.put(("data", epoch_buffer))

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def compute_alpha_power(data, fs, fmin=8, fmax=12):
    f, pxx = welch(data, fs=fs, nperseg=256)
    return np.sum(pxx[(f >= fmin) & (f <= fmax)])

# ---------------------------------------------------------------------
# GUI Classes (FrequencyWindow, TopographyWindow, EEGGUI)
# ---------------------------------------------------------------------
# ... [KEEP SAME GUI CODE AS ORIGINAL, shortened for brevity] ...

if __name__ == "__main__":
    app = EEGGUI()
    app.mainloop()
