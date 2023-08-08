"""
File: RawNet2.py
Version: 1.0
Author: Soumya Shaw
Last Edited: August 08, 2023 by Soumya Shaw
"""

# Module Imports
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchaudio.functional import compute_deltas

from dfadetect.datasets import lfcc, load_directory_split_train_test, mfcc
from dfadetect.models.raw_net2 import RawNet
from dfadetect.trainer import GDTrainer, GMMTrainer
from dfadetect.utils import set_seed
from experiment_config import RAW_NET_CONFIG, feature_kwargs

LOGGER = logging.getLogger()

def init_logger(log_file):
    LOGGER.setLevel(logging.INFO)

    # create file handler
    fh = logging.FileHandler(log_file)

    # create console handler
    ch = logging.StreamHandler()

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

def save_model(model: torch.nn.Module, model_dir: Union[Path, str], name: str) -> None:
    # Save model

    # Create model directory
    model_class = "raw_net"
    full_model_dir = Path(f"{model_dir}/{model_class}/{name}")

    # Create model directory if it does not exist
    if not full_model_dir.exists():
        full_model_dir.mkdir(parents=True)

    torch.save(model.state_dict(),
               f"{full_model_dir}/ckpt.pth")

def train_raw_net(
        real_training_distribution: Union[Path, str],
        fake_training_distributions: List[Union[Path, str]],
        batch_size: int,
        epochs: int,
        device: str,
        model_dir: Optional[str] = None,
        test_size: float = 0.2,
) -> None:

    LOGGER.info("Loading data...")

    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        real_training_distribution,
        None,
        None,
        test_size,
        pad=True,
        label=1,
    )

    # Train fake models
    for current in fake_training_distributions:
        LOGGER.info(f"Training {current}")
        print(f"Training {current}")
        fake_dataset_train, _ = load_directory_split_train_test(
            current,
            None,
            None,
            test_size,
            pad=True,
            label=0,
        )

        current_model = RawNet(deepcopy(RAW_NET_CONFIG), device).to(device)
        data_train = ConcatDataset([real_dataset_train, fake_dataset_train])
        LOGGER.info(
            f"Training rawnet model on {len(data_train)} audio files.")

        current_model = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs={
                "lr": 0.0001,
                "weight_decay": 0.0001,
            }
        ).train(
            dataset=data_train,
            model=current_model,
            test_len=test_size,
        )

        if model_dir is not None:
            save_model(current_model, model_dir, str(
                current).strip("/").replace("/", "_"))

def main(args):
    # Fix a seed for reproducibility
    set_seed(42)

    # Initialize logger
    init_logger("experiments_rawnet.log")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set feature function (mfcc or lfcc)
    feature_fn = mfcc

    # Parse fake directories
    """base_dir = Path(args.FAKE)
    fake_dirs = []
    for path in base_dir.iterdir():
        if path.is_dir():
            if "jsut" in str(path) or "conformer" in str(path):
                continue

            fake_dirs.append(path.absolute())

    if len(fake_dirs) == 0:
        fake_dirs = [base_dir]"""

    model_dir_path = f"{args.ckpt}"
    model_dir_path += f"/{'mfcc'}"
    model_dir = Path(model_dir_path)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)

    # Train models
    train_raw_net(
        real_training_distribution=args.REAL,
        fake_training_distributions=[Path(args.FAKE)],
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir if not args.debug else None,  # do not save debug models
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "REAL", help="Directory containing real data.", type=str)
    parser.add_argument(
        "FAKE", help="Directory containing fake data.", type=str)

    default_k = 128
    parser.add_argument(
        "--clusters", "-k", help=f"The amount of clusters to learn (default: {default_k}).", type=int, default=default_k)

    default_batch_size = 8
    parser.add_argument(
        "--batch_size", "-b", help=f"Batch size (default: {default_batch_size}).", type=int, default=default_batch_size)

    default_epochs = 5
    parser.add_argument(
        "--epochs", "-e", help=f"Epochs (default: {default_epochs}).", type=int, default=default_epochs)

    default_model_dir = "trained_models"

    parser.add_argument(
        "--ckpt", help=f"Checkpoint directory (default: {default_model_dir}).", type=str, default=default_model_dir)
    
    parser.add_argument(
        "--raw_net", help="Train raw net version?", action="store_true")
    
    parser.add_argument(
        "--lfcc", "-l", help="Use LFCC instead of MFCC?", action="store_true")
    
    parser.add_argument(
        "--debug", "-d", help="Only use minimal amount of files?", action="store_true")
    
    parser.add_argument(
        "--verbose", "-v", help="Display debug information?", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
