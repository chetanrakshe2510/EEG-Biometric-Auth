import h5py


def count_epochs_by_class(hdf5_path):
    """
    Count number of epochs per class from an HDF5 dataset.

    Expected structure:
      /class_<label>/Run_<id>/epoch_<n>/features

    Args:
        hdf5_path (str): Path to the HDF5 dataset.

    Returns:
        counts (dict): Mapping of class label -> number of epochs.
        total (int): Total number of epochs across all classes.
    """
    counts = {}
    with h5py.File(hdf5_path, 'r') as f:
        for class_group in f.keys():
            label = class_group.replace('class_', '')
            grp = f[class_group]
            epoch_count = 0
            for run_group in grp.values():
                for epoch_key in run_group.keys():
                    if epoch_key.startswith('epoch_'):
                        epoch_count += 1
            counts[label] = epoch_count
    return counts, sum(counts.values())


if __name__ == "__main__":
    # Update this to your actual HDF5 filename
    hdf5_path = r"G:\Publication and confernece\Workj_2\Results\all_subjects_merged_new_full_epochs.h5"

    counts, total = count_epochs_by_class(hdf5_path)
    print("\n=== OFFLINE CLASS BALANCE (from HDF5) ===")
    for label, cnt in counts.items():
        print(f"  Class {label}: {cnt} epochs ({cnt/total:.1%})")
