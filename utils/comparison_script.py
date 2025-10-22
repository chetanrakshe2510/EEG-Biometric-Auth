import sys
from pathlib import Path
import numpy as np
import mne
from joblib import load

# Ensure project directory is in path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import real-time and offline modules
import test_gui_rf              # Real-time pipeline
import updated_feature_extraction_global_baseline  # Offline pipeline

# Aliases for clarity
preproc_rt = test_gui_rf.preprocess_raw_data
extract_rt = test_gui_rf.extract_all_features
preproc_off = updated_feature_extraction_global_baseline.preprocess_raw_data
extract_off = updated_feature_extraction_global_baseline.extract_all_features

# Parameters
RAW_PATH = "test_raw_data.npy"
FS = 500
WINDOW_SEC = 2
OVERLAP = 0.5
WIN_SAMP = int(WINDOW_SEC * FS)
STEP = int(WIN_SAMP * (1 - OVERLAP))

# Load test raw data (saved from real-time outlet)
data = np.load(RAW_PATH)  # shape (n_channels, n_samples)

# Load transformers
transformer_rt = load("new_power_transformer.pkl")
transformer_off = load("new_power_transformer.pkl")


def run_pipeline(preproc_fn, extract_fn, transformer, data, is_offline=False):
    """
    Run full pipeline: preprocessing, baseline (if offline), feature extraction, transform.

    Args:
        preproc_fn (func): Preprocessing function.
        extract_fn (func): Feature extraction function.
        transformer: Scikit-learn transformer.
        data (np.ndarray): EEG data (n_channels, n_samples).
        is_offline (bool): Whether to apply offline baseline correction.

    Returns:
        X (np.ndarray): Raw feature matrix.
        X_t (np.ndarray): Transformed feature matrix.
    """
    ch_names = [f"Ch{idx+1}" for idx in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=FS, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Preprocess
    raw = preproc_fn(raw, l_freq=1.0, h_freq=40.0, target_fs=FS)
    arr, _ = raw[:, :]

    # Offline baseline correction
    if is_offline:
        arr = arr - arr.mean(axis=1, keepdims=True)

    # Epoching & feature extraction
    feats = []
    for start in range(0, arr.shape[1] - WIN_SAMP + 1, STEP):
        epoch = arr[:, start:start + WIN_SAMP]
        vec = np.hstack([extract_fn(epoch[ch], FS) for ch in range(epoch.shape[0])])
        feats.append(vec)
    X = np.vstack(feats)

    # Transform
    X_t = transformer.transform(X)
    return X, X_t


if __name__ == "__main__":
    # Run real-time pipeline
    X_rt, X_rt_t = run_pipeline(preproc_rt, extract_rt, transformer_rt, data, is_offline=False)

    # Run offline pipeline
    X_off, X_off_t = run_pipeline(preproc_off, extract_off, transformer_off, data, is_offline=True)

    def summary(name, X):
        print(f"\n=== {name} ===")
        print(f"mean={X.mean():.4f}, std={X.std():.4f}, min={X.min():.4f}, max={X.max():.4f}")

    summary("Real-time RAW", X_rt)
    summary("Real-time Transformed", X_rt_t)
    summary("Offline RAW", X_off)
    summary("Offline Transformed", X_off_t)

    # Correlation between transformed features
    corr = np.corrcoef(X_off_t.ravel(), X_rt_t.ravel())[0, 1]
    print(f"\nGlobal correlation (transformed features): {corr:.4f}")
