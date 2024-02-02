import os
import argparse
from pathlib import Path

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Lightning MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to train with')
    parser.add_argument('--epochs', type=int, default=31, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    return args

def main(args):
    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
