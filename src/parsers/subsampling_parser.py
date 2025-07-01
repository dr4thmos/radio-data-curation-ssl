import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--clusters_id", type=str)
    parser.add_argument("--target_size", type=int, default=None, help="Numero di campioni")
    return parser.parse_args()