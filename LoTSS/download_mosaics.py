import os
import requests

# Base URL
base_url = "https://lofar-surveys.org/"

# Lista di percorsi relativi da scaricare
file_paths = [
    "public/DR2/mosaics/{}/mosaic-blanked.fits",
    #"public/DR2/mosaics/{}/mosaic.rms.fits",
    #"public/DR2/mosaics/{}/mosaic.resid.fits",
    #"public/DR2/mosaics/{}/mosaic-weights.fits",
    "public/DR2/mosaics/{}/mosaic.pybdsmmask.fits",
    #"public/DR2/mosaics/{}/low-mosaic-blanked.fits",
    #"public/DR2/mosaics/{}/low-mosaic-weights.fits",
    "public/DR2/mosaics/{}/fits_headers.tar",
    "public/DR2/mosaics/{}/mosaic-blanked.png",
]

# Nome del file contenente gli ID
list_file = "to_download.txt"

# Directory di destinazione per i file scaricati
output_dir = "mosaics"
os.makedirs(output_dir, exist_ok=True)

def download_file(url, dest):
    """Scarica un file da una URL e lo salva nella destinazione specificata."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Scaricato: {dest}")
    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download di {url}: {e}")

def main():
    # Legge gli ID da list.txt
    try:
        with open(list_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {list_file} non trovato.")
        return

    # Itera su ogni ID e scarica i file corrispondenti
    for mosaic_id in ids:
        for file_path in file_paths:
            file_url = base_url + file_path.format(mosaic_id)
            file_name = os.path.basename(file_url)
            dest_path = os.path.join(output_dir, mosaic_id, file_name)
            
            # Crea directory per l'ID se non esiste
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Scarica il file
            download_file(file_url, dest_path)

if __name__ == "__main__":
    main()
