import json

import torch
import pandas as pd
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.Datasets.preprocess import Preprocessing
from src.models.Model import Complete_Autoencoder
from src.Train.trainer import Trainer
from src.utils import set_seed

torch.set_num_threads(8)

# ----------------------------------------------------------
# CONFIGURATION FILE
# ----------------------------------------------------------

with open("configs/classification_config.yaml", "r") as file:
    config = yaml.safe_load(file)

set_seed(config.get("seed", 42))

# ----------------------------------------------------------
# DATASET LOADING AND PREPROCESSING
# ----------------------------------------------------------

# Load raw data
rawdata = pd.read_csv("data/creditcard.csv")

# Preprocess:
#   - get_smote_dataset() applica SMOTE solo sul train set per bilanciare le classi
#     (il test set rimane con la distribuzione reale per una valutazione corretta)
#   - Le label sono restituite one-hot (N, 2), compatibili con CrossEntropyLoss
preprocess = Preprocessing(rawdata, drop_time=config.get("drop_time", False))

use_smote = config.get("use_smote", True)
class_weight = None

if use_smote:
    # SMOTE: bilancia il training set sinteticamente (50/50 fraud/normal)
    X_train, X_test, y_train, y_test = preprocess.get_smote_dataset(
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
    )
    print(f"Train set: {X_train.shape[0]} campioni (dopo SMOTE, ~50/50)")
else:
    # No SMOTE: dataset originale sbilanciato + Weighted CrossEntropy
    # I class weights compensano lo sbilanciamento penalizzando di più gli errori sul fraud
    X_train, X_test, y_train, y_test = preprocess.get_dataset(
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
    )
    weights = preprocess.get_class_weights(
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
        verbose=True,
    )
    class_weight = torch.tensor([weights[0], weights[1]], dtype=torch.float32)
    print(f"Train set: {X_train.shape[0]} campioni (originale, sbilanciato)")
    print(f"Class weights → Normal: {weights[0]:.4f}, Fraud: {weights[1]:.4f}")

print(f"Test  set: {X_test.shape[0]} campioni")

# ----------------------------------------------------------
# DATALOADERS
# ----------------------------------------------------------

batch_size = config["training"]["batch_size"]

NUM_WORKERS = 0

train_loader = DataLoader(
    dataset=TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

test_loader = DataLoader(
    dataset=TensorDataset(X_test, y_test),
    batch_size=batch_size,
    shuffle=False,   # shuffle=False sul test per riproducibilità
    num_workers=NUM_WORKERS,
)

# ----------------------------------------------------------
# MODEL AND TRAINING
# ----------------------------------------------------------

classifier = Complete_Autoencoder(config=config)
trainer = Trainer(model=classifier, config=config, class_weight=class_weight)

# ----------------------------------------------------------
# TRAINING PHASE
# ----------------------------------------------------------

print("Starting classifier training...")

# val_labels: etichette intere (0/1) per il calcolo del F1 a ogni epoca
# y_test è one-hot (N, 2) → argmax riporta a (N,) int
val_labels = torch.argmax(y_test, dim=1).numpy()

results = trainer.fit(train_loader=train_loader, val_loader=test_loader, val_labels=val_labels)

# ----------------------------------------------------------
# SAVE TRAINING HISTORY
# ----------------------------------------------------------

history_path = "histories/training_history_classifier.json"
print(f"Saving training history to {history_path}...")
with open(history_path, "w") as f:
    json.dump(results, f, indent=4)

print("Done. Checkpoint saved to:",
      f"checkpoints/{config['paths']['checkpoint_name']}")
