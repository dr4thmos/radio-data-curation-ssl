import h5py

# Path to the HDF5 file
hdf5_file_path = "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_resnet18-variant__9e41ae1d/features_data_fast.h5"

# Open the HDF5 file in read mode
with h5py.File(hdf5_file_path, "r") as hdf:
    # List all groups and datasets in the file
    print("Keys in the HDF5 file:", list(hdf.keys()))
    
    # Access a specific dataset (replace 'dataset_name' with the actual key)
    dataset_name = list(hdf.keys())[0]  # Example: Access the first dataset
    data = hdf[dataset_name][:]
    
    # Print dataset shape and content
    print(f"Dataset '{dataset_name}' shape:", data.shape)
    print(f"Dataset '{dataset_name}' content (first 5 rows):\n", data[:5])