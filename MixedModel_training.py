import pandas as pd
import torch
import yaml
import json
from torch.utils.data import TensorDataset, DataLoader

from src.models.Autoencoder import FCDecoder, LSTM_Encoder, FraudAutoencoder
from src.Train.trainer import Trainer
from src.Datasets.preprocess import Preprocessing
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator


# ----------------------------------------------------------
# CONFIGURATION FILE
# ---------------------------------------------------------- 

# Read configuration file
with open("configs/config_LSTM.yaml", "r") as file:
    config = yaml.safe_load(file)


# ----------------------------------------------------------
# DATASET LOADING AND PREPROCESSING
# ---------------------------------------------------------- 

# Load dataset
rawdata = pd.read_csv("data/creditcard.csv")

# Preprocess
preprocess = Preprocessing(df=rawdata, drop_time=True)

seq_cfg = config.get("sequence", {})
seq_len = seq_cfg.get("seq_len", 10)
stride = seq_cfg.get("stride", 1)
only_normal = seq_cfg.get("only_normal", True)
test_size = config.get("test_size", 0.2)

# Obtain 3D sequence datasets for LSTM Autoencoder
X_train_tensor, X_test_tensor, y_test_tensor = preprocess.get_sequence_dataset(
    seq_len=seq_len,
    stride=stride,
    test_size=test_size,
    autoencoder=True,
    only_normal=only_normal
)

# Transform the dataset into PyTorch DataLoaders
train_dataset = TensorDataset(X_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# ----------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------- 

batch_size = config.get("batch_size", 512)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True 
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size, 
    shuffle=False 
)

# ----------------------------------------------------------
# MODEL AND TRAINING CONFIGURATION 
# ---------------------------------------------------------- 

lstm_cfg = config.get("lstm_model", {})
input_dim = X_train_tensor.shape[2]  # Number of features per timestep

encoder = LSTM_Encoder(
    input_dim=input_dim,
    hidden_dim=lstm_cfg.get("hidden_dim", 64),
    latent_dim=lstm_cfg.get("latent_dim", 16),
    num_layers=lstm_cfg.get("num_layers", 2),
    dropout=lstm_cfg.get("dropout", 0.2)
)

decoder = FCDecoder(
    latent_dim=lstm_cfg.get("latent_dim", 16), 
    input_dim=input_dim
)

model = FraudAutoencoder(encoder, decoder)


# Training configuration and definition
trainer = Trainer(model=model, config=config)


# ----------------------------------------------------------
# TRAINING PHASE 
# ---------------------------------------------------------- 

print(f"Starting LSTM Autoencoder training (input_dim={input_dim}, seq_len={seq_len})...")
results = trainer.fit(
    train_loader=train_loader, 
    val_loader=test_loader
)

print("Saving training history...")
with open("training_history_LSTM.json", "w") as f:
    json.dump(results, f, indent=4)


# ----------------------------------------------------------
# EVALUATION PHASE 
# ---------------------------------------------------------- 
print("\n--- Starting Anomaly Detection Evaluation ---")

# 1. Define the device
device = torch.device(config.get("device", "cpu"))

# 2. Initialize the evaluator
evaluator = ReconstructionEvaluator(
    model=model, 
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
