import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.Autoencoder import Encoder, FraudAutoencoder
import os

# (Ensure your Encoder and FraudAutoencoder are defined here)

def prepare_data(csv_path="data/creditcard.csv", drop_time=True):
    """Loads dataset and splits into train (normal only) and test (mixed)."""
    df = pd.read_csv(csv_path)
    
    if drop_time and "Time" in df.columns:
        df = df.drop(columns=["Time"])
        
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Filter training data to ONLY contain normal transactions
    normal_mask_train = (y_train == 0)
    X_train_normal = X_train[normal_mask_train]
    
    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.float32)
    
    return train_tensor, test_tensor, test_labels


def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    input_dim = 29
    train_tensor, test_tensor, test_labels = prepare_data(drop_time=True)
    
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=256, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor, test_labels), batch_size=256, shuffle=False)
    
    # 2. Load Pre-trained Encoder
    encoder = Encoder(input_dim=input_dim)
    try:
        encoder.load_state_dict(torch.load("pretrained_tabular_encoder.pth", map_location=device, weights_only=True))
        print("Successfully loaded pre-trained encoder weights.")
    except FileNotFoundError:
        print("Pre-trained weights not found. Using randomly initialized encoder.")
        
    for param in encoder.parameters():
        param.requires_grad = False
        
    # 3. Initialize Autoencoder
    model = FraudAutoencoder(encoder_module=encoder, input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 4. Training Loop (Decoder Only)
    epochs = 300
    print("\n--- Starting Decoder Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}] - Reconstruction Loss (MSE): {avg_loss:.4f}")
        
    # 5. Evaluation Loop
    print("\n--- Starting Evaluation ---")
    model.eval()
    all_errors = []
    all_labels = []
    eval_criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)
            errors = eval_criterion(reconstructed, batch_x).mean(dim=1)
            
            all_errors.extend(errors.cpu().numpy())
            all_labels.extend(batch_y.numpy())
            
    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    
    # 6. Analyze Results & Find Threshold
    normal_errors = all_errors[all_labels == 0]
    fraud_errors = all_errors[all_labels == 1]
    
    print("\n--- Results ---")
    print(f"Average Normal Error: {normal_errors.mean():.4f}")
    print(f"Average Fraud Error:  {fraud_errors.mean():.4f}")
    
    auc = roc_auc_score(all_labels, all_errors)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Find the threshold that maximizes F1 score
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_errors)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    print(f"Optimal Anomaly Threshold (Max F1): {best_threshold:.4f}")
    
    # Make binary predictions
    preds = (all_errors > best_threshold).astype(int)

    # ---------------------------------------------------------
    # 7. PLOTTING
    # ---------------------------------------------------------
    
    output_dir = "results/Contrastive"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(all_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/ConfusionMatrix.png")
    

    # Plot 2: Reconstruction Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(normal_errors, bins=50, color='blue', alpha=0.5, label='Normal', stat='density', log_scale=(False, True))
    sns.histplot(fraud_errors, bins=50, color='red', alpha=0.5, label='Fraud', stat='density', log_scale=(False, True))
    plt.axvline(best_threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({best_threshold:.2f})')
    plt.title("Reconstruction Error Distribution (Log Scale Y)")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density (Log Scale)")
    plt.legend()
    # Limit x-axis to 99.5th percentile to prevent extreme outliers from squishing the plot
    plt.xlim(0, np.percentile(all_errors, 99.5)) 
    plt.savefig(f"{output_dir}/ReconstructionError.png")

    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_errors)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/ROC_Curve.png")


if __name__ == "__main__":
    train_and_evaluate()