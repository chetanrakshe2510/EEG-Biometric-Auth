# EEG Biometric Authentication System

A real-time **EEG-based biometric authentication system** that streams, preprocesses, extracts features, and classifies EEG signals using both **Machine Learning (Random Forest)** and **Template Matching (Mahalanobis Distance)**.

Built with:
- [MNE](https://mne.tools) for EEG processing
- [Lab Streaming Layer (LSL)](https://github.com/sccn/labstreaminglayer) for data streaming
- [Scikit-learn](https://scikit-learn.org/) for machine learning
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for the GUI

---

## 🚀 Features
- **Real-time EEG pipeline** with LSL
- Preprocessing: filtering, notch, ICA artifact removal, bad channel interpolation
- **Rich feature extraction**:
  - Statistical (skewness, kurtosis, Hjorth)
  - Frequency domain (band powers, Welch PSD)
  - Entropy (spectral, permutation)
  - Complexity (fractal dimensions: Katz, Higuchi, Hurst)
  - MFCCs
  - EMD-based entropy
- **Two recognition pipelines**:
  - **Random Forest Classifier** (offline trained, real-time predictions)
  - **Mahalanobis Distance** (template-based matching)
- **Interactive GUI**:
  - Real-time EEG waveform viewer
  - Power Spectrum & Spectrogram window
  - Topographic map of alpha-band (8–12 Hz) power
- **Offline utilities**:
  - Dataset balance check
  - Offline model evaluation
  - Real-time vs offline consistency validation

---

## 📂 Project Structure
EEG-Biometric-Auth/
│
├── realtime/
│   ├── test_gui_rf.py             # Random Forest pipeline 
│   ├── test_gui_mahalanobis.py    # Mahalanobis pipeline 
│
├── offline/
│   ├── offline_model_test.py      # Evaluate offline models
│   ├── Feature_file_test.py       # Check dataset balance
│
├── utils/
│   ├── comparison_script.py       # Compare RT vs offline pipelines
│
├── models/                        # Pretrained models (.pkl, .npy)
├── data/                          # Example .vhdr or .npy test data
│
├── requirements.txt
├── README.md
└── LICENSE
