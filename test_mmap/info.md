# Proof of Concept: Accesso Efficiente ai Cutout

## Obiettivo

Questo script investiga e confronta diverse strategie di storage e accesso ai dati per risolvere due problemi critici dell'attuale pipeline:

1.  **Esplosione dello Storage:** La materializzazione di centinaia di migliaia di file `.npy` (i cutout) consuma uno spazio su disco enorme e ridondante.
2.  **Collo di Bottiglia I/O:** La gestione di un numero così elevato di piccoli file è intrinsecamente inefficiente per il filesystem.

L'ipotesi è che mantenere i dati in un unico, grande file-mosaico e accedere ai cutout "virtualmente" (on-the-fly) sia superiore in termini di spazio su disco, footprint di memoria e potenzialmente anche velocità di accesso.

## Metodologia

Lo script esegue un benchmark controllato:

1.  **Generazione Dati:** Viene creata un'immagine sintetica di grandi dimensioni (`5000x5000`) popolata da sorgenti gaussiane per simulare un mosaico astronomico. Viene generata anche una lista di coordinate per i cutout.
2.  **Strategie di Storage Confrontate:** L'immagine viene salvata in quattro formati diversi:
    *   **Baseline (File Individuali):** La strategia attuale. Ogni cutout viene salvato in un file `.npy` separato.
    *   **NumPy Memory-Mapping (`.dat`):** Un unico file binario su disco. L'accesso ai dati avviene "mappando" solo la porzione richiesta in memoria, senza caricare l'intero file.
    *   **HDF5 (`.h5`):** Un formato standard per dati scientifici, ottimizzato per lo slicing efficiente di grandi array su disco.
    *   **PyTorch Tensor (`.pt`):** Un unico file. L'accesso richiede il caricamento dell'intero tensore in memoria (usato come controllo negativo per l'efficienza della memoria).
3.  **Benchmark:** Per ogni strategia, lo script misura:
    *   **Spazio su Disco:** La dimensione totale dei file generati.
    *   **Tempo di Accesso:** Il tempo medio per leggere un singolo cutout.
    *   **Footprint di Memoria:** L'incremento di RAM richiesto per leggere un singolo cutout.
4.  **Integrazione:** Vengono fornite implementazioni di `torch.utils.data.Dataset` per dimostrare come le strategie più efficienti (HDF5 e file individuali) si integrino in un workflow di training.

## Risultati Attesi

Si prevede che le strategie **HDF5** e **NumPy mmap** mostrino un consumo di memoria per accesso drasticamente inferiore rispetto al caricamento completo di un tensore PyTorch e un'efficienza di storage enormemente superiore rispetto ai file individuali, pur mantenendo tempi di accesso competitivi.

## Come Eseguire

1.  Assicurati di aver installato le dipendenze richieste:
    ```bash
    pip install numpy torch h5py matplotlib memory_profiler psutil
    ```
2.  Esegui lo script dalla riga di comando:
    ```bash
    python test_mmap.py
    ```
3.  Lo script genererà diversi file `.png` nella stessa directory, mostrando i grafici delle performance e alcuni cutout di esempio.

## Conclusione

Questo esperimento fornisce le prove quantitative per giustificare una profonda revisione architettonica del modo in cui i dati dei cutout vengono gestiti nella pipeline principale. I risultati dimostrano la fattibilità e i benefici di un approccio basato su file singoli e accesso virtuale.