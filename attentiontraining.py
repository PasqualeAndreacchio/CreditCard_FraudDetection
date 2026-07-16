import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.Model import Complete_Autoencoder
from src.trainer import Trainer
from src.preprocess import Preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml


# ----------------------------------------------------------
# DATASET LOADING AND PREPROCESSING
# ---------------------------------------------------------- 


# Load dataset
rawdata = pd.read_csv("data/creditcard.csv")

# Preprocess
preprocess = Preprocessing(rawdata)
Xtrain_smote, Xtest, Ytrain_smote, Ytest = preprocess.get_smote_dataset(test_size=0.2)

# Transform the dataset into the right data type and format
train_dataset = TensorDataset(Xtrain_smote, Ytrain_smote)
test_dataset = TensorDataset(Xtest, Ytest)


# ----------------------------------------------------------
# DATALOADERS
# ---------------------------------------------------------- 


# Define train and test dataloaders
batch_size = 64

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


# Read configuration file
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Model configuration and definition
detector = Complete_Autoencoder(config=config)

# Training configuration and definition
trainer = Trainer(model=detector, config=config)
