import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutout_ids", nargs="+", type=str)
    return parser.parse_args()