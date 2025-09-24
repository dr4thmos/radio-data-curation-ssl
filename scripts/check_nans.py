import h5py
import numpy as np

file_path = "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_vit_small_patch14_reg4_dinov2lvd142m-variant__b27e541a/features_data_fast.h5"

with h5py.File(file_path, 'r') as f:
    features_b = f['features'][:]
    
    total_elements = features_b.size
    nan_count = np.isnan(features_b).sum()
    
    print(f"Analisi del file: {file_path}")
    print(f"Dimensioni dell'array: {features_b.shape}")
    print(f"Numero totale di valori: {total_elements}")
    print(f"Numero totale di valori NaN: {nan_count}")
    
    if nan_count > 0:
        percentage = (nan_count / total_elements) * 100
        print(f"Percentuale di NaN: {percentage:.4f}%")
        
        # Controlla quanti campioni (immagini) sono affetti
        samples_with_nan = np.any(np.isnan(features_b), axis=1).sum()
        total_samples = features_b.shape[0]
        print(f"Numero di campioni (immagini) con almeno un NaN: {samples_with_nan} su {total_samples}")