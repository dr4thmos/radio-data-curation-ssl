import h5py
import numpy as np

# Percorso del suo file finale
hdf5_file_path = "/leonardo_work/INA24_C5B09/radio-data-curation-ssl/outputs/features/features-model_leonardo_workINA24_C5B09dinov2andreamodelsdinov2-small-imgnetgrayscale-dinoloss-50e-2x6checkpoint_small_dino-variant__c9d31093/features_data.h5"

# Apriamo il file in modalità lettura ('r')
with h5py.File(hdf5_file_path, 'r') as h5f:
    # Accediamo ai dataset. Questi oggetti NON caricano i dati in memoria.
    features_dset = h5f['features']
    paths_dset = h5f['image_paths']
    
    print(f"Dataset 'features' shape: {features_dset.shape}")
    print(f"Dataset 'image_paths' shape: {paths_dset.shape}")
    
    # Esempio: caricare solo le prime 100 righe in un array NumPy
    first_100_features = features_dset[:100] 
    print(f"\nShape of first 100 features loaded into RAM: {first_100_features.shape}")
    
    # Esempio: caricare un singolo path
    path_at_index_500 = paths_dset[500]
    print(f"Path at index 500: {path_at_index_500}")
    
    # Esempio: caricare TUTTE le feature in un array NumPy (solo se ha abbastanza RAM!)
    all_features_in_ram = features_dset[:]
    print(all_features_in_ram.shape)
    
    # Esempio: caricare tutti i path
    #all_paths_in_ram = paths_dset[:]