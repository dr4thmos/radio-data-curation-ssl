import os
import time
import numpy as np
import h5py
import zarr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
import psutil
import gc

# Configurazione
NUM_MOSAICS = 50
MOSAIC_SIZE = 10000  # 10k pixels per lato
CUTOUT_SIZES = [128, 256]  # dimensioni dei cutout
BATCH_SIZE = 64
NUM_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "path/to/fits_files"  # sostituisci con il percorso ai tuoi file
# /leonardo_scratch/large/userexternal/tceccone/LoTSS/DR2/mosaic_name_list.txt
RESULTS_DIR = "benchmark_results"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Funzione per generare dati di test simulati (rimuovi questa parte se hai già i dati)
def generate_test_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print("Generazione dati di test...")
    for i in range(NUM_MOSAICS):
        # Crea un file FITS con dati casuali
        data = np.random.randn(MOSAIC_SIZE, MOSAIC_SIZE).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        filename = os.path.join(DATA_DIR, f"mosaic_{i:03d}.fits")
        hdu.writeto(filename, overwrite=True)
        
        # Crea anche versioni HDF5 e Zarr per il benchmark
        h5_filename = os.path.join(DATA_DIR, f"mosaic_{i:03d}.h5")
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('data', data=data, chunks=(1024, 1024))
            
        zarr_filename = os.path.join(DATA_DIR, f"mosaic_{i:03d}.zarr")
        z = zarr.open(zarr_filename, mode='w')
        z.create_dataset('data', data=data, chunks=(1024, 1024))
    
    print(f"Generati {NUM_MOSAICS} mosaici in {DATA_DIR}")

# Dataset di base per il training contrastivo
class ContrastiveDataset(Dataset):
    def __init__(self, cutout_size=256):
        self.cutout_size = cutout_size
        self.files = self._get_file_list()
        self.num_files = len(self.files)
    
    def _get_file_list(self):
        # Implementato nelle classi derivate
        pass
    
    def _get_random_cutout(self, idx):
        # Implementato nelle classi derivate
        pass
    
    def __len__(self):
        return self.num_files * 100  # 100 cutout per mosaico
    
    def __getitem__(self, idx):
        file_idx = idx % self.num_files
        
        # Prendi due cutout dallo stesso mosaico per il training contrastivo
        img1 = self._get_random_cutout(file_idx)
        img2 = self._get_random_cutout(file_idx)
        
        # Normalizza e converti in tensor
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        
        # Normalizzazione semplice (adatta per i tuoi dati se necessario)
        img1 = (img1 - img1.mean()) / (img1.std() + 1e-6)
        img2 = (img2 - img2.mean()) / (img2.std() + 1e-6)
        
        # Aggiungi dimensione del canale
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        return img1, img2

# Dataset con Numpy mmap
class MmapDataset(ContrastiveDataset):
    def __init__(self, cutout_size=256):
        super().__init__(cutout_size)
        self.mmaps = {}
    
    def _get_file_list(self):
        return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                if f.endswith('.fits')]
    
    def _load_mmap(self, file_path):
        if file_path not in self.mmaps:
            with fits.open(file_path, memmap=True) as hdul:
                data = hdul[0].data
                # Crea un mmap del file
                self.mmaps[file_path] = np.memmap(file_path, dtype=np.float32, mode='r',
                                                 shape=data.shape, offset=fits.open(file_path)[0]._data_offset)
        return self.mmaps[file_path]
    
    def _get_random_cutout(self, idx):
        file_path = self.files[idx]
        data = self._load_mmap(file_path)
        
        # Scegli coordinate casuali per il cutout
        max_x = MOSAIC_SIZE - self.cutout_size
        max_y = MOSAIC_SIZE - self.cutout_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        
        return data[y:y+self.cutout_size, x:x+self.cutout_size].copy()

# Dataset con HDF5
class HDF5Dataset(ContrastiveDataset):
    def __init__(self, cutout_size=256):
        super().__init__(cutout_size)
        self.h5_files = {}
    
    def _get_file_list(self):
        return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                if f.endswith('.h5')]
    
    def _get_h5_file(self, file_path):
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        return self.h5_files[file_path]
    
    def _get_random_cutout(self, idx):
        file_path = self.files[idx]
        h5_file = self._get_h5_file(file_path)
        data = h5_file['data']
        
        max_x = MOSAIC_SIZE - self.cutout_size
        max_y = MOSAIC_SIZE - self.cutout_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        
        return data[y:y+self.cutout_size, x:x+self.cutout_size]
    
    def __del__(self):
        # Chiudi tutti i file quando l'oggetto viene distrutto
        for f in self.h5_files.values():
            f.close()

# Dataset con Zarr
class ZarrDataset(ContrastiveDataset):
    def __init__(self, cutout_size=256):
        super().__init__(cutout_size)
        self.zarr_arrays = {}
    
    def _get_file_list(self):
        return [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
                if f.endswith('.zarr')]
    
    def _get_zarr_array(self, file_path):
        if file_path not in self.zarr_arrays:
            self.zarr_arrays[file_path] = zarr.open(file_path, mode='r')['data']
        return self.zarr_arrays[file_path]
    
    def _get_random_cutout(self, idx):
        file_path = self.files[idx]
        data = self._get_zarr_array(file_path)
        
        max_x = MOSAIC_SIZE - self.cutout_size
        max_y = MOSAIC_SIZE - self.cutout_size
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        
        return data[y:y+self.cutout_size, x:x+self.cutout_size]

# Modello semplice per l'addestramento contrastivo
class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def forward(self, x):
        h = self.encoder(x).squeeze(-1).squeeze(-1)
        z = self.projection(h)
        return nn.functional.normalize(z, dim=1)

# Funzione di perdita contrastiva (InfoNCE/NT-Xent)
def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    
    # Calcola la similarità coseno tra tutti i campioni
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t().contiguous()) / temperature
    
    # Maschera per rimuovere la similarità di un campione con se stesso
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    
    # Etichette positive
    positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0)
    
    # Maschera per rimuovere le diagonali
    mask = torch.ones_like(sim)
    mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    
    # Campioni negativi
    negative_samples = sim[mask.bool()].view(2*batch_size, -1)
    
    # InfoNCE loss
    logits = torch.cat([positive_samples.unsqueeze(1), negative_samples], dim=1)
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=DEVICE)
    
    return nn.CrossEntropyLoss()(logits, labels)

# Funzione per addestrare il modello
def train_model(dataloader, dataset_name, cutout_size):
    model = ContrastiveModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Metriche di benchmark
    batch_times = []
    memory_usage = []
    losses = []
    
    model.train()
    total_epochs = NUM_EPOCHS
    
    for epoch in range(total_epochs):
        epoch_loss = 0
        num_batches = 0
        
        print(f"\nEpoca {epoch+1}/{total_epochs} - {dataset_name} (cutout: {cutout_size})")
        
        # Loop di training con progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (img1, img2) in enumerate(progress_bar):
            start_time = time.time()
            
            # Sposta i dati sulla GPU
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            
            # Forward pass
            z1 = model(img1)
            z2 = model(img2)
            
            # Calcola la perdita
            loss = contrastive_loss(z1, z2)
            
            # Backward e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calcola il tempo trascorso per il batch
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Monitora l'utilizzo della memoria
            memory = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else psutil.Process().memory_info().rss / (1024 ** 3)
            memory_usage.append(memory)
            
            # Aggiorna la loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Aggiorna la progress bar
            progress_bar.set_postfix(loss=loss.item(), time=f"{batch_time:.3f}s", memory=f"{memory:.2f}GB")
            
            # Limita il numero di batch per il benchmark
            if batch_idx >= 50:
                break
        
        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)
        print(f"  Loss media: {avg_epoch_loss:.4f}")
        print(f"  Tempo medio per batch: {np.mean(batch_times[-num_batches:]):.3f}s")
        print(f"  Memoria media: {np.mean(memory_usage[-num_batches:]):.2f}GB")
    
    # Raccogli i risultati del benchmark
    results = {
        "dataset": dataset_name,
        "cutout_size": cutout_size,
        "avg_batch_time": np.mean(batch_times),
        "std_batch_time": np.std(batch_times),
        "max_memory": max(memory_usage),
        "avg_memory": np.mean(memory_usage),
        "final_loss": losses[-1]
    }
    
    # Pulisci la GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

# Funzione principale per eseguire tutti i benchmark
def run_benchmarks():
    results = []
    
    # Test con diverse dimensioni di cutout
    for cutout_size in CUTOUT_SIZES:
        # Benchmark Numpy mmap
        print(f"\n--- Benchmark Numpy mmap (cutout: {cutout_size}) ---")
        dataset = MmapDataset(cutout_size=cutout_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        result = train_model(dataloader, "Numpy mmap", cutout_size)
        results.append(result)
        
        # Libera memoria
        del dataset, dataloader
        gc.collect()
        
        # Benchmark HDF5
        print(f"\n--- Benchmark HDF5 (cutout: {cutout_size}) ---")
        dataset = HDF5Dataset(cutout_size=cutout_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        result = train_model(dataloader, "HDF5", cutout_size)
        results.append(result)
        
        # Libera memoria
        del dataset, dataloader
        gc.collect()
        
        # Benchmark Zarr
        print(f"\n--- Benchmark Zarr (cutout: {cutout_size}) ---")
        dataset = ZarrDataset(cutout_size=cutout_size)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
        result = train_model(dataloader, "Zarr", cutout_size)
        results.append(result)
        
        # Libera memoria
        del dataset, dataloader
        gc.collect()
    
    # Salva e visualizza i risultati
    save_and_plot_results(results)

# Funzione per salvare e visualizzare i risultati
def save_and_plot_results(results):
    # Salva i risultati in un file CSV
    import pandas as pd
    df = pd.DataFrame(results)
    csv_file = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nRisultati salvati in {csv_file}")
    
    # Visualizza i risultati
    plt.figure(figsize=(15, 10))
    
    # Plot dei tempi di batch
    plt.subplot(2, 2, 1)
    for cutout_size in CUTOUT_SIZES:
        df_filtered = df[df['cutout_size'] == cutout_size]
        plt.bar(df_filtered['dataset'], df_filtered['avg_batch_time'], yerr=df_filtered['std_batch_time'], alpha=0.7, label=f"Cutout {cutout_size}")
    plt.title('Tempo medio per batch')
    plt.ylabel('Tempo (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot dell'utilizzo di memoria
    plt.subplot(2, 2, 2)
    for cutout_size in CUTOUT_SIZES:
        df_filtered = df[df['cutout_size'] == cutout_size]
        plt.bar(df_filtered['dataset'], df_filtered['max_memory'], alpha=0.7, label=f"Cutout {cutout_size}")
    plt.title('Utilizzo massimo di memoria')
    plt.ylabel('Memoria (GB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot delle loss finali
    plt.subplot(2, 2, 3)
    for cutout_size in CUTOUT_SIZES:
        df_filtered = df[df['cutout_size'] == cutout_size]
        plt.bar(df_filtered['dataset'], df_filtered['final_loss'], alpha=0.7, label=f"Cutout {cutout_size}")
    plt.title('Loss finale')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Aggiungi riepilogo testuale
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = "RIEPILOGO BENCHMARK:\n\n"
    
    for result in results:
        summary_text += f"{result['dataset']} (cutout {result['cutout_size']}):\n"
        summary_text += f"  - Tempo batch: {result['avg_batch_time']:.3f}s ± {result['std_batch_time']:.3f}s\n"
        summary_text += f"  - Memoria: {result['avg_memory']:.2f}GB (max: {result['max_memory']:.2f}GB)\n"
        summary_text += f"  - Loss finale: {result['final_loss']:.4f}\n\n"
    
    plt.text(0, 1, summary_text, fontsize=10, verticalalignment='top')
    
    # Salva il grafico
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_plots.png"))
    plt.close()
    print(f"Grafici salvati in {os.path.join(RESULTS_DIR, 'benchmark_plots.png')}")

if __name__ == "__main__":
    # Genera dati di test se non esistono
    if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) < NUM_MOSAICS * 3:
        generate_test_data()
    
    # Esegui tutti i benchmark
    run_benchmarks()