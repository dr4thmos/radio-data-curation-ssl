import numpy as np
import torch
import h5py
import os
import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import psutil
import gc  # Import garbage collector

# Dimensioni dell'immagine e dei cutout
IMAGE_SIZE = (5000, 5000)  # Immagine abbastanza grande
CUTOUT_SIZE = (256, 256)  # Dimensione standard per i cutout
N_GAUSSIANS = 100  # Numero di gaussiane da generare
OVERLAP_PERCENT = 0.5  # Percentuale di overlap desiderata


# Funzione per generare una gaussiana 2D
def gaussian_2d(size, center, sigma):
    y, x = np.indices(size)
    y = y - center[0]
    x = x - center[1]
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


# Genera l'immagine con gaussiane e le coordinate dei cutout
def generate_image_with_gaussians(overlap_percent=0.0):  # Aggiunto overlap_percent
    # Inizializza l'immagine vuota
    image = np.zeros(IMAGE_SIZE, dtype=np.float32)

    # Lista per memorizzare le posizioni e i parametri delle gaussiane
    cutout_info = []

    # Genera N gaussiane con parametri casuali
    for i in range(N_GAUSSIANS):
        # Parametri casuali
        if overlap_percent > 0:
            # Calcola l'overlap in pixel
            overlap_x = int(CUTOUT_SIZE[1] * overlap_percent)
            overlap_y = int(CUTOUT_SIZE[0] * overlap_percent)

            # Definisci i limiti per il centro del cutout, tenendo conto dell'overlap
            min_x = CUTOUT_SIZE[1] // 2 - overlap_x
            max_x = IMAGE_SIZE[1] - CUTOUT_SIZE[1] // 2 + overlap_x
            min_y = CUTOUT_SIZE[0] // 2 - overlap_y
            max_y = IMAGE_SIZE[0] - CUTOUT_SIZE[0] // 2 + overlap_y

            center_x = np.random.randint(min_x, max_x)
            center_y = np.random.randint(min_y, max_y)
        else:
            center_y = np.random.randint(
                CUTOUT_SIZE[0] // 2, IMAGE_SIZE[0] - CUTOUT_SIZE[0] // 2
            )
            center_x = np.random.randint(
                CUTOUT_SIZE[1] // 2, IMAGE_SIZE[1] - CUTOUT_SIZE[1] // 2
            )

        sigma = np.random.uniform(10, 50)
        amplitude = np.random.uniform(0.5, 1.0)

        # Calcola angoli del cutout
        start_y = max(0, center_y - CUTOUT_SIZE[0] // 2)
        start_x = max(0, center_x - CUTOUT_SIZE[1] // 2)

        # Genera la gaussiana nel cutout
        local_center = (center_y - start_y, center_x - start_x)
        gaussian = gaussian_2d(CUTOUT_SIZE, local_center, sigma) * amplitude

        # Inserisci la gaussiana nell'immagine, gestendo i bordi
        end_y = min(start_y + CUTOUT_SIZE[0], IMAGE_SIZE[0])
        end_x = min(start_x + CUTOUT_SIZE[1], IMAGE_SIZE[1])
        
        # Adjust gaussian size if necessary
        gaussian_y_end = end_y - start_y
        gaussian_x_end = end_x - start_x
        
        image[start_y:end_y, start_x:end_x] += gaussian[:gaussian_y_end, :gaussian_x_end]

        # Memorizza le informazioni del cutout
        cutout_info.append(
            {
                "start_y": start_y,
                "start_x": start_x,
                "center_y": center_y - start_y,
                "center_x": center_x - start_x,
                "sigma": sigma,
                "amplitude": amplitude,
            }
        )

    return image, cutout_info


# Prepara i file per i test
def prepare_files(image):
    # 1. NumPy mmap
    mmap_filename = "big_image_mmap.dat"
    fp = np.memmap(mmap_filename, dtype=np.float32, mode="w+", shape=IMAGE_SIZE)
    fp[:] = image[:]
    fp.flush()

    # 2. PyTorch
    torch_filename = "big_image_torch.pt"
    torch_tensor = torch.tensor(image)
    torch.save(torch_tensor, torch_filename)

    # 3. HDF5
    h5_filename = "big_image_h5.h5"
    with h5py.File(h5_filename, "w") as f:
        f.create_dataset("image", data=image, chunks=CUTOUT_SIZE)

    return mmap_filename, torch_filename, h5_filename


# Prepara i singoli file numpy per ogni cutout
def prepare_individual_numpy_files(image, cutout_info):
    """Salva ogni cutout in un file numpy separato."""
    # Crea directory se non esiste
    directory = "cutouts_numpy"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Salva ogni cutout in un file separato
    filenames = []
    for i, cutout in enumerate(cutout_info):
        start_y = cutout["start_y"]
        start_x = cutout["start_x"]
        cutout_data = image[
            start_y : start_y + CUTOUT_SIZE[0], start_x : start_x + CUTOUT_SIZE[1]
        ]

        filename = os.path.join(directory, f"cutout_{i}.npy")
        np.save(filename, cutout_data)
        filenames.append(filename)

    return filenames


# Funzioni di accesso ai cutout
def get_cutout_numpy_mmap(mmap_filename, start_y, start_x, height, width):
    fp = np.memmap(mmap_filename, dtype=np.float32, mode="r", shape=IMAGE_SIZE)
    return fp[start_y : start_y + height, start_x : start_x + width].copy()


def get_cutout_torch(torch_filename, start_y, start_x, height, width):
    # Carica tutto il tensore e poi estrae il cutout (non è memory-mapped)
    tensor = torch.load(torch_filename)
    return tensor[start_y : start_y + height, start_x : start_x + width].numpy()


def get_cutout_h5(h5_filename, start_y, start_x, height, width):
    with h5py.File(h5_filename, "r") as f:
        return f["image"][start_y : start_y + height, start_x : start_x + width]


def get_cutout_individual_numpy(filename):
    """Carica un cutout da un singolo file numpy."""
    return np.load(filename)


# Funzione per misurare la memoria in modo più affidabile
def measure_memory_usage(func, *args):
    # Forza il garbage collector
    gc.collect()

    # Misura prima
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss

    # Esegui la funzione
    result = func(*args)

    # Misura dopo
    memory_after = process.memory_info().rss

    return result, (memory_after - memory_before) / (1024 * 1024)  # MB


# Test di velocità e memoria con misurazione più accurata
def run_comparison_improved(mmap_filename, torch_filename, h5_filename, cutout_info, individual_numpy_files):
    results = {}
    memory_usages = {}
    access_times = []

    # Numero di accessi da testare
    n_accesses = min(20, len(cutout_info))  # Limitiamo per essere più veloci

    # Seleziona alcuni cutout da accedere
    cutouts_to_access = cutout_info[:n_accesses]

    # Test NumPy mmap
    numpy_times = []
    numpy_memories = []
    for i, cutout in enumerate(cutouts_to_access):
        start_time = time.time()
        _, memory_used = measure_memory_usage(
            get_cutout_numpy_mmap,
            mmap_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )
        end_time = time.time()
        numpy_times.append(end_time - start_time)
        numpy_memories.append(memory_used)

    results["numpy_mmap"] = {
        "time": np.mean(numpy_times),
        "memory": np.mean(numpy_memories),
        "times": numpy_times,
        "memories": numpy_memories,
    }

    # Test PyTorch (caricamento completo)
    torch_times = []
    torch_memories = []
    for i, cutout in enumerate(cutouts_to_access):
        start_time = time.time()
        _, memory_used = measure_memory_usage(
            get_cutout_torch,
            torch_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )
        end_time = time.time()
        torch_times.append(end_time - start_time)
        torch_memories.append(memory_used)

    results["torch"] = {
        "time": np.mean(torch_times),
        "memory": np.mean(torch_memories),
        "times": torch_times,
        "memories": torch_memories,
    }

    # Test HDF5
    h5_times = []
    h5_memories = []
    for i, cutout in enumerate(cutouts_to_access):
        start_time = time.time()
        _, memory_used = measure_memory_usage(
            get_cutout_h5,
            h5_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )
        end_time = time.time()
        h5_times.append(end_time - start_time)
        h5_memories.append(memory_used)

    results["h5"] = {
        "time": np.mean(h5_times),
        "memory": np.mean(h5_memories),
        "times": h5_times,
        "memories": h5_memories,
    }

    # Test singoli file NumPy
    individual_numpy_times = []
    individual_numpy_memories = []
    for i, cutout in enumerate(cutouts_to_access):
        if i < len(individual_numpy_files):
            start_time = time.time()
            _, memory_used = measure_memory_usage(
                get_cutout_individual_numpy, individual_numpy_files[i]
            )
            end_time = time.time()
            individual_numpy_times.append(end_time - start_time)
            individual_numpy_memories.append(memory_used)

    results["individual_numpy"] = {
        "time": np.mean(individual_numpy_times),
        "memory": np.mean(individual_numpy_memories),
        "times": individual_numpy_times,
        "memories": individual_numpy_memories,
    }

    access_times = list(range(n_accesses))
    memory_usages = {
        "numpy_mmap": numpy_memories,
        "torch": torch_memories,
        "h5": h5_memories,
        "individual_numpy": individual_numpy_memories,
    }

    return results, access_times, memory_usages


# Visualizzazione dei cutout e salvataggio delle immagini
def visualize_and_save_cutouts(mmap_filename, h5_filename, torch_filename, cutout_info, individual_numpy_files, num_samples=3):
    samples = np.random.choice(range(len(cutout_info)), num_samples, replace=False)

    for i, idx in enumerate(samples):
        cutout = cutout_info[idx]

        # Ottieni cutout da numpy mmap
        mmap_cutout = get_cutout_numpy_mmap(
            mmap_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )

        # Ottieni cutout da h5
        h5_cutout = get_cutout_h5(
            h5_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )

        # Ottieni cutout da torch
        torch_cutout = get_cutout_torch(
            torch_filename,
            cutout["start_y"],
            cutout["start_x"],
            CUTOUT_SIZE[0],
            CUTOUT_SIZE[1],
        )

        # Ottieni cutout da file numpy individuale
        individual_numpy_cutout = get_cutout_individual_numpy(
            individual_numpy_files[idx]
        )

        # Crea la figura
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        im0 = axes[0].imshow(mmap_cutout, cmap="viridis")
        axes[0].set_title(f"NumPy mmap - sigma: {cutout['sigma']:.2f}")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(h5_cutout, cmap="viridis")
        axes[1].set_title(f"HDF5 - sigma: {cutout['sigma']:.2f}")
        plt.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(torch_cutout, cmap="viridis")
        axes[2].set_title(f"PyTorch - sigma: {cutout['sigma']:.2f}")
        plt.colorbar(im2, ax=axes[2])

        im3 = axes[3].imshow(individual_numpy_cutout, cmap="viridis")
        axes[3].set_title(f"Individual NumPy - sigma: {cutout['sigma']:.2f}")
        plt.colorbar(im3, ax=axes[3])

        plt.tight_layout()
        plt.savefig(f"cutout_comparison_{i}.png", dpi=150)
        plt.close()

        print(f"Immagine {i} salvata come cutout_comparison_{i}.png")

        # Verifica differenze numeriche
        mmap_h5_diff = np.abs(mmap_cutout - h5_cutout).max()
        mmap_torch_diff = np.abs(mmap_cutout - torch_cutout).max()
        h5_torch_diff = np.abs(h5_cutout - torch_cutout).max()
        individual_numpy_diff = np.abs(mmap_cutout - individual_numpy_cutout).max()

        print(f"Differenza max NumPy-HDF5: {mmap_h5_diff}")
        print(f"Differenza max NumPy-PyTorch: {mmap_torch_diff}")
        print(f"Differenza max HDF5-PyTorch: {h5_torch_diff}")
        print(f"Differenza max NumPy-Individual: {individual_numpy_diff}")


# Plot di prestazioni
def plot_performance(results, access_times, memory_usages):
    # Plot del tempo medio di accesso
    plt.figure(figsize=(14, 6))
    methods = list(results.keys())
    times = [results[method]["time"] for method in methods]

    colors = ["blue", "orange", "green", "red"]  # Un colore per ogni metodo

    plt.subplot(1, 2, 1)
    plt.bar(methods, times, color=colors[: len(methods)])
    plt.title("Tempo medio di accesso")
    plt.ylabel("Tempo (secondi)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Plot della memoria media utilizzata
    memories = [results[method]["memory"] for method in methods]

    plt.subplot(1, 2, 2)
    plt.bar(methods, memories, color=colors[: len(methods)])
    plt.title("Memoria media utilizzata")
    plt.ylabel("Memoria (MB)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("performance_comparison.png", dpi=150)
    plt.close()

    # Plot dell'uso di memoria nel tempo
    plt.figure(figsize=(10, 6))
    markers = ["o", "s", "^", "D"]  # Marker diversi per ogni metodo
    for i, (method, memory_vals) in enumerate(memory_usages.items()):
        plt.plot(
            access_times[: len(memory_vals)],
            memory_vals,
            label=method,
            marker=markers[i % len(markers)],
        )

    plt.title("Utilizzo di memoria per accesso")
    plt.xlabel("Numero di accesso")
    plt.ylabel("Memoria (MB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("memory_usage_over_time.png", dpi=150)
    plt.close()

    # Traccia dei tempi di accesso
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        plt.plot(
            access_times[: len(results[method]["times"])],
            results[method]["times"],
            label=method,
            marker=markers[i % len(markers)],
        )

    plt.title("Tempi di accesso per ogni estrazione")
    plt.xlabel("Numero di accesso")
    plt.ylabel("Tempo (secondi)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("access_times.png", dpi=150)
    plt.close()


# Crea una classe Dataset di PyTorch per HDF5
class H5CutoutDataset(torch.utils.data.Dataset):
    def __init__(self, h5_filename, cutout_info):
        self.h5_filename = h5_filename
        self.cutout_info = cutout_info

    def __len__(self):
        return len(self.cutout_info)

    def __getitem__(self, idx):
        cutout = self.cutout_info[idx]

        with h5py.File(self.h5_filename, "r") as f:
            data = f["image"][
                cutout["start_y"] : cutout["start_y"] + CUTOUT_SIZE[0],
                cutout["start_x"] : cutout["start_x"] + CUTOUT_SIZE[1],
            ]

        # Simuliamo un label usando sigma come feature da predire
        label = torch.tensor([cutout["sigma"]], dtype=torch.float32)

        return torch.tensor(data, dtype=torch.float32), label


# Crea una classe Dataset di PyTorch per i singoli file NumPy
class IndividualNumpyDataset(torch.utils.data.Dataset):
    def __init__(self, numpy_filenames, cutout_info):
        self.numpy_filenames = numpy_filenames
        self.cutout_info = cutout_info

    def __len__(self):
        return len(self.numpy_filenames)

    def __getitem__(self, idx):
        cutout = self.cutout_info[idx]
        # Carica il file numpy corrispondente
        data = np.load(self.numpy_filenames[idx])

        # Simuliamo un label usando sigma come feature da predire
        label = torch.tensor([cutout["sigma"]], dtype=torch.float32)

        return torch.tensor(data, dtype=torch.float32), label


# Visualizzazione dell'immagine completa e di alcuni cutout
def visualize_full_image(image, cutout_info, num_samples=3):
    # Visualizza una versione downscalata dell'immagine completa
    scale_factor = 10  # Riduzione di 10 volte
    small_img = image[::scale_factor, ::scale_factor]

    plt.figure(figsize=(10, 10))
    plt.imshow(small_img, cmap="viridis")
    plt.title(f"Immagine completa (downscalata {scale_factor}x)")
    plt.colorbar()
    plt.savefig("full_image_downscaled.png", dpi=150)
    plt.close()

    # Visualizza alcuni cutout originali
    samples = np.random.choice(range(len(cutout_info)), num_samples, replace=False)

    for i, idx in enumerate(samples):
        cutout = cutout_info[idx]
        cutout_data = image[
            cutout["start_y"] : cutout["start_y"] + CUTOUT_SIZE[0],
            cutout["start_x"] : cutout["start_x"] + CUTOUT_SIZE[1],
        ]

        plt.figure(figsize=(8, 8))
        plt.imshow(cutout_data, cmap="viridis")
        plt.title(
            f'Cutout {i+1} - Sigma: {cutout["sigma"]:.2f}, Ampiezza: {cutout["amplitude"]:.2f}'
        )
        plt.colorbar()
        plt.savefig(f"original_cutout_{i}.png", dpi=150)
        plt.close()

        print(f"Cutout originale {i} salvato come original_cutout_{i}.png")


# Modifica della funzione main per includere il test di overlap
def main():
    overlap_percentages = [0.0, 0.25, 0.50, 0.75]  # Testiamo diversi livelli di overlap

    for overlap_percent in overlap_percentages:
        print(f"\nTest con overlap del {overlap_percent * 100}%")
        print("Generazione dell'immagine con gaussiane...")
        image, cutout_info = generate_image_with_gaussians(
            overlap_percent
        )  # Passa il parametro
        print(f"Immagine generata con {len(cutout_info)} gaussiane")

        # Visualizza e salva l'immagine completa e alcuni cutout originali
        print("Salvataggio dell'immagine completa e di alcuni cutout originali...")
        visualize_full_image(image, cutout_info)

        print("Preparazione dei file di test...")
        mmap_filename, torch_filename, h5_filename = prepare_files(image)

        print("Preparazione dei singoli file numpy per i cutout...")
        individual_numpy_files = prepare_individual_numpy_files(image, cutout_info)

        print("Test delle dimensioni dei file...")
        mmap_size = os.path.getsize(mmap_filename) / (1024 * 1024)  # MB
        torch_size = os.path.getsize(torch_filename) / (1024 * 1024)  # MB
        h5_size = os.path.getsize(h5_filename) / (1024 * 1024)  # MB

        # Calcola la dimensione totale dei file numpy individuali
        individual_numpy_size = (
            sum(os.path.getsize(f) for f in individual_numpy_files) / (1024 * 1024)
        )  # MB

        print(f"Dimensione file NumPy mmap: {mmap_size:.2f} MB")
        print(f"Dimensione file PyTorch: {torch_size:.2f} MB")
        print(f"Dimensione file HDF5: {h5_size:.2f} MB")
        print(
            f"Dimensione totale dei {len(individual_numpy_files)} file NumPy individuali: {individual_numpy_size:.2f} MB"
        )
        print(
            f"Dimensione media per singolo file NumPy: {individual_numpy_size/len(individual_numpy_files):.4f} MB"
        )

        print("Esecuzione dei test di prestazioni migliorati..")
        results, access_times, memory_usages = run_comparison_improved(
            mmap_filename, torch_filename, h5_filename, cutout_info, individual_numpy_files
        )
        print("Risultati dei test di prestazioni:")
        print(results)

        print("Visualizzazione dei risultati...")
        plot_performance(results, access_times, memory_usages)

        print("Visualizzazione e salvataggio dei cutout...")
        visualize_and_save_cutouts(
            mmap_filename, h5_filename, torch_filename, cutout_info, individual_numpy_files
        )

        # Crea e testa i DataLoader di PyTorch
        print("Creazione e test dei DataLoader di PyTorch...")
        h5_dataset = H5CutoutDataset(h5_filename, cutout_info)
        h5_dataloader = torch.utils.data.DataLoader(h5_dataset, batch_size=4, shuffle=True)

        individual_numpy_dataset = IndividualNumpyDataset(
            individual_numpy_files, cutout_info
        )
        individual_numpy_dataloader = torch.utils.data.DataLoader(
            individual_numpy_dataset, batch_size=4, shuffle=True
        )

        # Esempio di utilizzo dei DataLoader (solo un paio di batch per non allungare troppo l'esecuzione)
        print("Esempio di utilizzo HDF5 DataLoader:")
        for i, (data, labels) in enumerate(h5_dataloader):
            print(f"Batch {i+1}: Dati shape: {data.shape}, Labels shape: {labels.shape}")
            if i >= 1:  # Limita a 2 batch
                break

        print("Esempio di utilizzo Individual NumPy DataLoader:")
        for i, (data, labels) in enumerate(individual_numpy_dataloader):
            print(f"Batch {i+1}: Dati shape: {data.shape}, Labels shape: {labels.shape}")
            if i >= 1:  # Limita a 2 batch
                break

        # Pulisci i file creati
        print("Pulizia dei file temporanei...")
        os.remove(mmap_filename)
        os.remove(torch_filename)
        os.remove(h5_filename)
        # Rimuovi la directory con i file numpy
        import shutil

        shutil.rmtree("cutouts_numpy")
        print("Fine del test.")

if __name__ == "__main__":
    main()
