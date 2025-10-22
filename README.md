# EEG Biometric Authentication System

A real-time **EEG-based biometric authentication system** that streams, preprocesses, extracts features, and classifies EEG signals using both **Machine Learning (Random Forest)** and **Template Matching (Mahalanobis Distance)**. The system supports **real-time recognition** via a Tkinter-based GUI and includes utilities for **offline analysis** and **pipeline validation**.

Built with:
- [MNE](https://mne.tools) for EEG preprocessing
- [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) for EEG data streaming
- [Scikit-learn](https://scikit-learn.org/) for machine learning
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI visualization
- [Librosa](https://librosa.org/), [PyEMD](https://github.com/laszukdawid/PyEMD), [AntroPy](https://github.com/raphaelvallat/antropy), and [OrdPy](https://github.com/arthur-tacca/ordpy) for feature extraction

---

## 🚀 Key Features
- **Real-time EEG pipeline** with LSL streaming
- **Preprocessing**: resampling, filtering, notch removal, ICA-based artifact rejection, and bad channel interpolation
- **Comprehensive Feature Extraction**:
  - Statistical (skewness, kurtosis, Hjorth parameters)
  - Frequency domain (band powers via Welch PSD)
  - Entropy measures (spectral, permutation)
  - Complexity measures (Hurst exponent, Katz FD, Higuchi FD)
  - Mel-Frequency Cepstral Coefficients (MFCCs)
  - Empirical Mode Decomposition (EMD) based entropy
- **Classification Pipelines**:
  - Random Forest Classifier (offline trained, real-time predictions)
  - Mahalanobis Distance (template-based subject matching)
- **Interactive GUI**:
  - Live EEG waveform visualization
  - Power Spectrum Density & Spectrogram window
  - Alpha-band (8–12 Hz) topographic brain maps
- **Offline Utilities**:
  - Dataset class balance checker
  - Offline model evaluation (accuracy, reports, confusion matrix)
  - Real-time vs offline feature pipeline comparison

---

## 📂 Project Structure
```
EEG-Biometric-Auth/
│
├── realtime/
│   ├── test_gui_rf.py              # Random Forest real-time pipeline
│   ├── test_gui_mahalanobis.py     # Mahalanobis real-time pipeline
│
├── offline/
│   ├── offline_model_test.py       # Offline model evaluation
│   ├── feature_file_test.py        # Dataset class balance checker
│
├── utils/
│   ├── comparison_script.py        # Compare RT vs offline feature pipelines
│
├── models/                         # Pretrained models (.pkl, .npy)
├── data/                           # Example EEG data (.vhdr, .npy, .h5)
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/EEG-Biometric-Auth.git
cd EEG-Biometric-Auth
pip install -r requirements.txt
```

---

## 🖥️ Usage
### Run Real-time GUI (Random Forest)
```bash
python realtime/test_gui_rf.py
```

### Run Real-time GUI (Mahalanobis)
```bash
python realtime/test_gui_mahalanobis.py
```

### Offline Model Evaluation
```bash
python offline/offline_model_test.py
```

### Dataset Balance Check
```bash
python offline/feature_file_test.py
```

### Compare Real-time vs Offline Pipelines
```bash
python utils/comparison_script.py
```





## 📜 License
This project is licensed under the MIT License.

---

## 🙌 Acknowledgments
- [MNE](https://mne.tools) community for EEG processing
- [SCCN](https://github.com/sccn/labstreaminglayer) for Lab Streaming Layer
- [Scikit-learn](https://scikit-learn.org/) developers
- Open-source contributors for feature extraction libraries
