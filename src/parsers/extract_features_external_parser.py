# Contenuto per parsers/extract_features_parser.py

import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Estrae features da cutout di immagini.")
    
    # --- Argomenti per l'Input ---
    parser.add_argument('--input_folder', type=str, required=True, 
                        help='Percorso alla cartella contenente i cutout .npy e il file info.')
    parser.add_argument('--info_json_name', type=str, default='info.json', 
                        help='Nome del file JSON con la lista dei cutout (default: info.json).')

    # --- Argomenti per il Modello ---
    parser.add_argument('--model_type', type=str, required=True, choices=['cecconello_ssl', 'andrea_dino', 'thingsvision'],
                        help="Tipo di modello da usare.")
    parser.add_argument('--model_name', type=str, required=True, 
                        help="Nome del modello (es. 'resnet18', 'facebook/dino-vits16').")
    parser.add_argument('--ckpt_path', type=str, 
                        help="Percorso al checkpoint del modello (richiesto per 'cecconello_ssl').")
    parser.add_argument('--module_name', type=str, required=True,
                        help="Nome del layer da cui estrarre le features (es. 'avgpool').")
    parser.add_argument('--source', type=str, default='torchvision',
                        help="Sorgente per i modelli 'thingsvision' (es. 'torchvision', 'timm').")
    parser.add_argument('--model_parameters', nargs='*', default=[],
                        help="Parametri aggiuntivi per il modello (es. 'variant=BIT_S_R101x1').")
    
    # --- Argomenti per il Preprocessing ---
    parser.add_argument('--model_input_channels', type=int, default=3, choices=[1, 3],
                        help="Numero di canali di input attesi dal modello (1 o 3).")
    parser.add_argument('--normalization', type=str, default='minmax',
                        help="Tipo di normalizzazione da applicare (es. 'minmax', 'imagenet').")
    parser.add_argument('--resize', type=int, default=224,
                        help="Dimensione a cui ridimensionare le immagini.")

    # --- Argomenti per l'Esecuzione ---
    parser.add_argument('--batch_size', type=int, default=1024,
                        help="Dimensione del batch per l'estrazione.")
    parser.add_argument('--cuda_devices', type=str, default="0",
                        help="ID dei dispositivi CUDA da usare (es. '0' o '0,1').")
    parser.add_argument('--test_mode', action='store_true',
                        help="Attiva la modalità test, processando solo pochi batch.")
    parser.add_argument('--test_batches', type=int, default=3,
                        help="Numero di batch da processare in modalità test.")
    
    args = parser.parse_args()
    return args