import os
import argparse
from pathlib import Path
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import lightning as L  # Note it is no longer pytorch_lightning!
from src.model import MNISTModel


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
    parser.add_argument('--dataset_path', type=str, default="DATASET",
                        help='Point to where the dataset is')
    args = parser.parse_args()
    return args


def split_train(dataset_full):
    # use 20% of training data for validation
    train_set_size = int(len(dataset_full) * 0.8)
    valid_set_size = len(dataset_full) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(dataset_full, [train_set_size, valid_set_size], generator=seed)
    return train_set, valid_set


def main(args):
    model = MNISTModel()
    # setup data
    transform = transforms.ToTensor()
    dataset_path = os.getenv(args.dataset_path)
    dataset_full = MNIST(root=dataset_path,
                         train=True, download=True, transform=transform)
    dataset_train, dataset_val = split_train(dataset_full)
    loader_train = data.DataLoader(dataset_train)
    loader_val = data.DataLoader(dataset_val)

    dataset_test = MNIST(root=dataset_path,
                         train=False, download=True, transform=transform)
    loader_test = data.DataLoader(dataset_test)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = L.Trainer(devices=args.gpus,
                        accelerator="gpu",
                        strategy="deepspeed_stage_2",
                        precision=16,
                        max_epochs=args.epochs)

    trainer.fit(model=model,
                train_dataloaders=loader_train, valid_loader=loader_val)

    trainer.test(model, dataloaders=loader_test)

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
