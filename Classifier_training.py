import argparse
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.Datasets.preprocess import Preprocessing
from src.models.Classifier import FraudDetectionMLP
from src.Train.trainer import Trainer


# Training function using Trainer
def train_classifier(config_path: str = "configs/classification_config.yaml") -> None:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    seed        = config.get("seed", 42)
    test_size   = config.get("test_size", 0.2)
    val_size    = config.get("val_size", 0.1)
    batch_size  = config["training"].get("batch_size", 512)
    drop_time   = config.get("drop_time", True)
    use_smote   = config.get("use_smote", True)

    # Build the dataset via Preprocessing (scaling + stratified split)
    data = pd.read_csv("data/creditcard.csv")
    preprocessor = Preprocessing(data, drop_time=drop_time)

    if use_smote:
        # SMOTE is applied to the training set only; val and test are left untouched
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_smote_dataset(
            test_size=test_size, val_size=val_size, random_state=seed
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_dataset(
            test_size=test_size, val_size=val_size, random_state=seed, autoencoder=False
        )

    # Wrap tensors in DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    # Build the model
    input_dim = X_train.shape[1]
    model = FraudDetectionMLP(input_dim=input_dim)

    # val_labels are needed by Trainer to compute F1/AUPRC on the validation set
    val_labels = y_val.squeeze().numpy()

    # Delegate the full training lifecycle to Trainer
    trainer = Trainer(model=model, config=config)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, val_labels=val_labels)


# Main function to parse arguments and launch training
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the FraudDetectionMLP classifier"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classification_config.yaml",
        help="Path to the YAML configuration file (default: configs/classification_config.yaml)",
    )
    args = parser.parse_args()
    train_classifier(config_path=args.config)


if __name__ == "__main__":
    main()
