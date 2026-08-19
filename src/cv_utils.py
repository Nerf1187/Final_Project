import copy
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import autocast, GradScaler
from util import run_inference


def get_cv_folds(dataframe, n_splits=5, random_state=42):
    """
    Generate K-fold splits while keeping patient groups together.
    """
    df = dataframe.copy()
    if 'patient' not in df.columns:
        df['patient'] = df['PatientID'].str.extract(r'(P_\d+)')
    
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, val_idx in splitter.split(df, df['pathology'], df['patient']):
        folds.append((df.iloc[train_idx], df.iloc[val_idx]))
    return folds

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, early_stopping_patience=10):
    """
    Trains a model for a single fold.
    """
    scaler = GradScaler()
    best_val_loss = np.inf
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    loss_history = []
    val_loss_history = []
    val_auc_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * images.size(0)
            
            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        val_labels, val_probs, val_loss = run_inference(model, val_loader, criterion, device)
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_loss)
        
        print(f"--- Epoch {epoch + 1} Completed | Train Loss: {epoch_loss / len(train_loader.dataset):.4f} "
              f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} ---")
        
        loss_history.append(epoch_loss / len(train_loader.dataset))
        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= early_stopping_patience > 0:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    model.load_state_dict(best_model_weights)
    return model, {"loss": best_val_loss, "auc": max(val_auc_history)}
