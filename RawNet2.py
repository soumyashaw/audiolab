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
from dfadetect.models.gaussian_mixture_model import (GMMEM, GMMDescent,
                                                     flatten_dataset)
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

    print(model_dir_path)

    if args.raw_net:
        train_raw_net(
            real_training_distribution=args.REAL,
            fake_training_distributions=[Path(args.FAKE)],
            amount_to_use=args.amount if not args.debug else 100,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_dir=model_dir if not args.debug else None,  # do not save debug models
        )

    """else:
        train_models(
            real_training_distribution=args.REAL,
            fake_training_distributions=fake_dirs,
            amount_to_use=args.amount if not args.debug else 100,
            feature_fn=feature_fn,
            feature_kwargs=feature_kwargs(args.lfcc),
            clusters=args.clusters,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            retraining=args.retraining,
            use_em=args.use_em,
            model_dir=model_dir if not args.debug else None,  # do not save debug models
        )"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "REAL", help="Directory containing real data.", type=str)
    parser.add_argument(
        "FAKE", help="Directory containing fake data.", type=str)

    default_amount = None
    parser.add_argument(
        "--amount", "-a", help=f"Amount of files to load from each directory (default: {default_amount} - all).", type=int, default=default_amount)

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
