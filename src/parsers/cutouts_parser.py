import argparse

def get_parser_sliding_window():
    parser = argparse.ArgumentParser(description="Crea ritagli da mosaici LoTSS usando una finestra scorrevole.")
    parser.add_argument("--window_size", type=int, default=256, help="Dimensione della finestra per i ritagli.")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap tra le finestre di ritaglio")
    parser.add_argument("--mosaics_path", type=str, help="Percorso alla cartella contenente i mosaici LoTSS.")
    return parser.parse_args()

def get_parser_mask():
    parser = argparse.ArgumentParser(description="Crea ritagli da mosaici LoTSS usando maschere delle sorgenti")
    return parser.parse_args()