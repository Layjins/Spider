import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=0, type=int)
    args = parser.parse_args()
    return args

