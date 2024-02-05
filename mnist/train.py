import os
import argparse
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import lightning as L  # Note it is no longer pytorch_lightning!
from lightning.pytorch import loggers as pl_loggers

from src.model import MNISTModel


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Lightning MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to train with')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--checkpoint_path', default="CHECKPOINTPATH", type=str,
                        help='For Saving the models checkpoints')
    parser.add_argument('--dataset_path', default="DATASETPATH", type=str,
                        help='Point to where the dataset is')
    parser.add_argument('--log_path', default="LOGPATH", type=str,
                        help='Point to where the logs will go')
    args = parser.parse_args()
    return args


def split_train(dataset_full, seed):
    # use 20% of training data for validation
    train_set_size = int(len(dataset_full) * 0.8)
    valid_set_size = len(dataset_full) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(seed)
    train_set, valid_set = data.random_split(dataset_full, [train_set_size, valid_set_size], generator=seed)
    return train_set, valid_set


def main(args):
    model = MNISTModel(lr=args.lr, seed=args.seed)
    # setup data
    transform = transforms.ToTensor()
    try:
        dataset_path = os.getenv(args.dataset_path)
    except Exception as e:
        print(e)
        dataset_path = args.dataset_path
    dataset_full = MNIST(root=dataset_path,
                         train=True, download=True, transform=transform)
    dataset_train, dataset_val = split_train(dataset_full, args.seed)
    loader_train = data.DataLoader(dataset_train)
    loader_val = data.DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=7)

    dataset_test = MNIST(root=dataset_path,
                         train=False, download=True, transform=transform)
    loader_test = data.DataLoader(dataset_test, batch_size=args.batch_size, pin_memory=True, num_workers=7)

    # logger
    try:
        log_path = os.getenv(args.log_path)
    except Exception as e:
        print(e)
        log_path = args.log_path
    print(log_path)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_path)

    # train the model
    trainer = L.Trainer(
        enable_model_summary=True,
        inference_mode=True,
        accelerator="gpu",
        strategy="ddp",
        sync_batchnorm=True,
        precision="16-mixed",
        devices=args.gpus,
        max_epochs=args.epochs,
        logger=logger)

    trainer.fit(model=model,
                train_dataloaders=loader_train,
                val_dataloaders=loader_val,
                ckpt_path=os.getenv(args.checkpoint_path),)

    trainer.test(model, dataloaders=loader_test)

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)
