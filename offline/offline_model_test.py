import h5py
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(hdf5_path):
    """
    Load features and labels from an HDF5 dataset.

    Expected structure:
      /class_<label>/Run_<id>/epoch_<n>/features

    Returns:
        X (np.ndarray): Feature matrix (n_samples, n_features)
        y (np.ndarray): Labels array (n_samples,)
    """
    X, y = [], []
    with h5py.File(hdf5_path, 'r') as f:
        for class_group in f.values():
            label = class_group.name.split('/')[-1].replace('class_', '')
            for run_group in class_group.values():
                for epoch_group in run_group.values():
                    feat = epoch_group['features'][()]
                    X.append(feat.flatten())
                    y.append(label)
    return np.vstack(X), np.array(y)


if __name__ == "__main__":
    # Update this path to your dataset
    hdf5_path = r"G:\Publication and confernece\Workj_2\Results\all_subjects_merged_new_full_epochs.h5"

    print("Loading offline data…")
    X_off, y_off = load_data(hdf5_path)

    transformer = load("new_power_transformer.pkl")
    model = load("new_logistic_regression_model.pkl")

    print("Transforming offline data…")
    X_off_t = transformer.transform(X_off)

    print("Predicting offline…")
    y_pred = model.predict(X_off_t)

    print("\n=== Accuracy ===")
    print(f"{accuracy_score(y_off, y_pred):.2%}")

    print("\n=== Classification Report ===")
    print(classification_report(y_off, y_pred, digits=3))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_off, y_pred, labels=model.classes_)
    print(cm)
