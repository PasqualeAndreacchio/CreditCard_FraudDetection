import argparse
import json
import logging
import os
import sys
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader

from src.models.LSTM_Autoencoder import LSTM_Autoencoder
from src.Train.trainer import Trainer
from src.Datasets.preprocess import Preprocessing
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator


def main() -> None:
    # ----------------------------------------------------------
    # CONFIGURATION FILE AND ARGUMENT PARSING
    # ---------------------------------------------------------- 
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder for Credit Card Fraud Detection.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_LSTM.yaml",
        help="Path to configuration YAML file (default: configs/config_LSTM.yaml)"
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=None,
        help="Number of DataLoader worker processes for parallel data loading (overrides config)"
    )
    parser.add_argument(
        "--drop_time",
        action="store_true",
        default=None,
        help="Drop 'Time' feature from dataset (overrides config)"
    )
    parser.add_argument(
        "--no_drop_time",
        action="store_false",
        dest="drop_time",
        help="Keep 'Time' feature in dataset"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run evaluation directly on the best saved checkpoint"
    )
    args = parser.parse_args()

    # Read configuration file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # ----------------------------------------------------------
    # LOGGING SETUP
    # ----------------------------------------------------------
    log_dir = config.get("paths", {}).get("log_dir", "logs/lstm")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "trainer.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Output will be saved to: {log_file_path}")

    # ----------------------------------------------------------
    # DATASET LOADING AND PREPROCESSING
    # ---------------------------------------------------------- 
    seq_cfg = config.get("sequence", {})
    drop_time = args.drop_time if args.drop_time is not None else seq_cfg.get("drop_time", True)

    logger.info(f"Loading raw dataset (drop_time={drop_time})...")
    rawdata = pd.read_csv("data/creditcard.csv")

    # Preprocess
    preprocess = Preprocessing(rawdata, drop_time=drop_time)

    seq_len = seq_cfg.get("seq_len", 10)
    stride = seq_cfg.get("stride", 1)
    only_normal = seq_cfg.get("only_normal", True)
    test_size = config.get("test_size", 0.2)
    val_size = config.get("val_size", 0.0)

    # Obtain 3D sequence datasets for LSTM Autoencoder
    if val_size and val_size > 0:
        X_train_tensor, X_val_tensor, X_test_tensor, y_val_tensor, y_test_tensor = preprocess.get_sequence_dataset(
            seq_len=seq_len,
            stride=stride,
            test_size=test_size,
            val_size=val_size,
            autoencoder=True,
            only_normal=only_normal
        )
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    else:
        X_train_tensor, X_test_tensor, y_test_tensor = preprocess.get_sequence_dataset(
            seq_len=seq_len,
            stride=stride,
            test_size=test_size,
            autoencoder=True,
            only_normal=only_normal
        )
        val_dataset = None

    # Transform the dataset into PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # ----------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------- 
    batch_size = config.get("batch_size", 512)
    num_workers = args.num_workers if args.num_workers is not None else config.get("num_workers", 0)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
    else:
        val_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    # ----------------------------------------------------------
    # MODEL AND TRAINING CONFIGURATION 
    # ---------------------------------------------------------- 
    lstm_cfg = config.get("lstm_model", {})
    input_dim = X_train_tensor.shape[2]  # Number of features per timestep

    detector = LSTM_Autoencoder(
        input_dim=input_dim,
        hidden_dim=lstm_cfg.get("hidden_dim", 64),
        latent_dim=lstm_cfg.get("latent_dim", 16),
        num_layers=lstm_cfg.get("num_layers", 2),
        dropout=lstm_cfg.get("dropout", 0.2)
    )

    # Training configuration and definition
    trainer = Trainer(model=detector, config=config)

    # ----------------------------------------------------------
    # TRAINING PHASE 
    # ---------------------------------------------------------- 
    if not args.eval_only:
        val_labels_np = y_val_tensor.cpu().numpy() if (val_size and val_size > 0) else None
        logger.info(f"Starting LSTM Autoencoder training (input_dim={input_dim}, seq_len={seq_len}, num_workers={num_workers})...")
        results = trainer.fit(
            train_loader=train_loader, 
            val_loader=val_loader,
            val_labels=val_labels_np
        )

        logger.info("Saving training history...")
        with open("training_history_LSTM.json", "w") as f:
            json.dump(results, f, indent=4)
    else:
        logger.info("Skipping training phase (--eval_only specified).")

    # ----------------------------------------------------------
    # EVALUATION PHASE 
    # ---------------------------------------------------------- 
    logger.info("--- Starting Anomaly Detection Evaluation ---")

    # Load best checkpoint before running evaluation
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "checkpoints/lstm")
    checkpoint_name = config.get("paths", {}).get("checkpoint_name", "lstm_autoencoder_best.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    if os.path.exists(best_ckpt_path):
        logger.info(f"Loading best model checkpoint for evaluation: {best_ckpt_path}")
        trainer.load_checkpoint(best_ckpt_path)

    # 1. Define the device
    device = torch.device(config.get("device", "cpu"))

    # 2. Initialize the evaluator
    evaluator = ReconstructionEvaluator(
        model=detector, 
        config=config, 
        device=device
    )

    # 3. Extract the ground-truth labels as a NumPy array
    labels_np = y_test_tensor.cpu().numpy()

    # 4. Run the complete evaluation suite
    evaluator.evaluate(
        loader=test_loader, 
        labels=labels_np
    )


if __name__ == "__main__":
    main()