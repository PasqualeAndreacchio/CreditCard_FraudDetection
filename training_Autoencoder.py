import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.Autoencoder import FraudAutoencoder
from src.Train.trainer import Trainer
from src.Datasets.preprocess import Preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import json
from src.Evaluation.reconstruction_evaluator import ReconstructionEvaluator

# ----------------------------------------------------------
# CONFIGURATION FILE
# ---------------------------------------------------------- 


# Read configuration file
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


# ----------------------------------------------------------
# DATASET LOADING AND PREPROCESSING
# ---------------------------------------------------------- 


# Load dataset
rawdata = pd.read_csv("data/creditcard.csv")

# Preprocess
preprocess = Preprocessing(rawdata)
X_train_tensor, X_test_tensor, y_test_tensor = preprocess.get_dataset(test_size=0.2, autoencoder=True)

# Transform the dataset into the right data type and format
train_dataset = X_train_tensor
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# ----------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------- 


# Define train and test dataloaders
batch_size = config.get("batch_size")

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True 
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size, 
    shuffle=True 
)

# ----------------------------------------------------------
# MODEL AND TRAINING CONFIGURATION 
# ---------------------------------------------------------- 


# Model configuration and definition
detector = FraudAutoencoder(input_dim=X_train_tensor.shape[1])


# Training configuration and definition
trainer = Trainer(model=detector, config=config)


# ----------------------------------------------------------
# TRAINING PHASE 
# ---------------------------------------------------------- 

results = trainer.fit(
    train_loader=train_loader, 
    val_loader=test_loader
)

print("Saving training history...")
with open("training_history.json", "w") as f:
    json.dump(results, f, indent=4)


# ----------------------------------------------------------
# EVALUATION PHASE 
# ---------------------------------------------------------- 
print("\n--- Starting Anomaly Detection Evaluation ---")

# 1. Define the device (make sure it matches where your model is)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Initialize the evaluator
evaluator = ReconstructionEvaluator(
    model=detector, 
    config=config, 
    device=device
)

# 3. Extract the ground-truth labels as a NumPy array 
# (The evaluator uses scikit-learn, which requires NumPy arrays instead of PyTorch Tensors)
labels_np = y_test_tensor.cpu().numpy()

# 4. Run the complete evaluation suite
evaluator.evaluate(
    loader=test_loader, 
    labels=labels_np
)