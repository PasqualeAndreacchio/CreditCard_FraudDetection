import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.models.Classifier import FraudDetectionMLP

# 1. Load and prepare data
data = pd.read_csv("data/creditcard.csv")

# Drop the 'Time' column as it is usually not helpful without heavy feature engineering
X = data.drop(['Class', 'Time'], axis=1).values 
y = data['Class'].values

# 2. Split FIRST to prevent data leakage!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale the data (StandardScaler is crucial for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # Scale test set using train set parameters

# 4. Apply SMOTE ONLY to the training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Convert to PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train_res)
y_train_tensor = torch.FloatTensor(y_train_res).unsqueeze(1) # Shape: (batch, 1)

# 6. Create DataLoader
dataset = TensorDataset(X_train_tensor, y_train_tensor)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 7. Device, Model, Loss, Optimizer setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudDetectionMLP(input_dim=X_train.shape[1]).to(device)

# BCEWithLogitsLoss is more numerically stable than adding a Sigmoid to the model
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 8. Training Loop
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (samples, labels) in enumerate(loader):
        # Move data to the same device as the model
        samples, labels = samples.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(samples)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
            
    print(f"--- Epoch {epoch+1} Average Loss: {total_loss / len(loader):.4f} ---")