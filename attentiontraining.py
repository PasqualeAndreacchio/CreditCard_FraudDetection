import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.models.complete_autoencoder import EncoderConfig, DecoderConfig, Complete_Autoencoder
from src.preprocess import Preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm

# Read raw data
rawdata = pd.read_csv("data/creditcard.csv")

# Apply SMOTE to the dataset
preprocess = Preprocessing(rawdata)
Xtrain_smote, Xtest_smote, Ytrain_smote, Ytest_smote = preprocess.get_smote_dataset(test_size=0.2)

# Transform the dataset into the right data type and format
Xtrain = torch.tensor(Xtrain_smote.values, dtype=torch.float32)
Ytrain = torch.tensor(Ytrain_smote.values, dtype=torch.float32).unsqueeze(-1)
train_dataset = TensorDataset(Xtrain, Ytrain)

# Define dataloader
batch_size = 64
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True 
)

# Define the model
<<<<<<< HEAD
d_embed = 32
encoder = Encoder(d_embed=d_embed, d_ff=16, num_heads=4)
classifier = Classifier(d_embed=d_embed, dropout=0.1)
detector = CreditCardFraudDetector(encoder, classifier, d_embed)
=======
d_embed = 48
encoder_config = EncoderConfig(d_embed=d_embed, d_ff=32, num_heads=4, dropout=0.1)
decoder_config = DecoderConfig(latent_dim=48, output_dim=1)

detector = Complete_Autoencoder(encoder_config, decoder_config)
>>>>>>> complete_model

#--------------------------------------------------------------------
# TRAINING PROCESS
#--------------------------------------------------------------------

# Global variables
epochs = 25
losses = []

# loss and optimizer
optimizer = torch.optim.Adam(detector.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for i in range(epochs):

    print(f"\n--- Epoch {i+1}/{epochs} ---")
    running_loss = 0.0
    
    loop = tqdm(train_loader, total=len(train_loader), desc="Training")

    for trans, trans_type in loop:
        pred = detector(trans)
        curr_loss = criterion(pred, trans_type)
        running_loss+=curr_loss.item()

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()
        loop.set_postfix(loss=curr_loss.item())

    epoch_loss = running_loss/len(train_loader)
    print("epoch loss: ", epoch_loss)
    losses.append(epoch_loss)

# Save the network weights

torch.save(detector.state_dict(), "detector_weights.pth")

# Losses plot 

plt.figure(figsize=(10, 5))
# Use .plot instead of .scatter for a smooth line
plt.plot(losses, label="Epoch Loss", color="royalblue", alpha=0.7) 

plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.savefig("training_loss_complete_model.png")

