import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.amp import autocast, GradScaler

from util import run_inference


def get_cv_folds(dataframe: pd.DataFrame,
                 n_splits: int = 5,
                 random_state: int = 42,
                 verbose: bool = True) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate K stratified folds while keeping every patient's images inside a single fold.

    A patient contributes several crops to CBIS-DDSM (left/right, CC/MLO, one per abnormality),
    so splitting on rows would place the same lesion on both sides of a fold boundary and inflate
    the scores. Grouping on the bare ``P_#####`` number keeps them together while
    :class:`~sklearn.model_selection.StratifiedGroupKFold` holds the benign/malignant ratio
    roughly constant across folds.

    :param dataframe: Dataframe to split. Must contain 'PatientID' and 'pathology' columns.
    :type dataframe: pandas.DataFrame
    :param n_splits: Number of folds. Defaults to 5.
    :type n_splits: int
    :param random_state: Seed controlling the shuffling of patient groups. Defaults to 42.
    :type random_state: int
    :param verbose: If True, print each fold's size and malignant fraction, and assert that no
        patient is shared between a fold's train and validation halves. Defaults to True.
    :type verbose: bool
    :return: A list of (train_df, val_df) tuples, one per fold. The added 'patient' column is
        retained on each split.
    :rtype: list[tuple[pandas.DataFrame, pandas.DataFrame]]
    """

    df = dataframe.copy()
    if 'patient' not in df.columns:
        df['patient'] = df['PatientID'].str.extract(r'(P_\d+)')

    # A failed regex match yields NaN, which StratifiedGroupKFold reports as an opaque error
    # much later, so fail loudly here instead.
    missing = df['patient'].isna().sum()
    if missing:
        raise ValueError(f"Could not extract a patient number from {missing} PatientID value(s).")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, val_idx in splitter.split(df, df['pathology'], df['patient']):
        folds.append((df.iloc[train_idx], df.iloc[val_idx]))

    if verbose:
        for i, (train_df, val_df) in enumerate(folds):
            overlap = set(train_df['patient']) & set(val_df['patient'])
            assert not overlap, (f"Patient leakage in fold {i + 1}: "
                                 f"{len(overlap)} patient(s) in both train and val.")
            print(f"Fold {i + 1}: train {len(train_df):>5} rows / "
                  f"{train_df['patient'].nunique():>4} patients "
                  f"(malignant {(train_df['pathology'] == 'MALIGNANT').mean():.3f})  |  "
                  f"val {len(val_df):>4} rows / {val_df['patient'].nunique():>4} patients "
                  f"(malignant {(val_df['pathology'] == 'MALIGNANT').mean():.3f})")
        print(f"No patient is shared between train and val in any of the {len(folds)} folds.")

    return folds


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
               device, early_stopping_patience: int = 10, verbose: bool = True) -> tuple:
    """
    Train a model on one fold and restore the weights from its best validation epoch.

    The model, optimizer and scheduler are all mutated in place, so each fold must be given
    **freshly constructed** objects. Reusing them across folds would carry weights trained on
    fold *k*'s validation patients into fold *k+1*, along with the previous fold's Adam moments
    and decayed learning rate.

    :param model: Model to train. Modified in place and returned holding the best weights.
    :param train_loader: DataLoader over this fold's training rows.
    :param val_loader: DataLoader over this fold's held-out rows.
    :param criterion: Loss function.
    :param optimizer: Optimizer over ``model``'s trainable parameters.
    :param scheduler: LR scheduler stepped on the validation loss each epoch.
    :param num_epochs: Maximum number of epochs to train for.
    :type num_epochs: int
    :param device: Device to train on.
    :param early_stopping_patience: Stop after this many epochs without a validation-loss
        improvement. Pass 0 to disable. Defaults to 10.
    :type early_stopping_patience: int
    :param verbose: If True, print per-batch and per-epoch progress. Defaults to True.
    :type verbose: bool
    :return: A (model, history) tuple. ``history`` holds the full 'train_loss', 'val_loss' and
        'val_auc' curves plus the selected epoch and the loss *and* AUC measured at that same
        epoch, so the two reported numbers describe the one checkpoint that is kept.
    :rtype: tuple
    """

    device = torch.device(device)
    device_type = device.type

    # GradScaler is a no-op outside CUDA; keying it and autocast off the requested device
    # rather than cuda availability keeps an explicit device='cpu' run working on a CUDA box.
    scaler = GradScaler(device_type, enabled=(device_type == 'cuda'))

    best_val_loss = np.inf
    best_epoch = 0
    best_epoch_auc = float('nan')
    best_model_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    train_loss_history = []
    val_loss_history = []
    val_auc_history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type=device_type, enabled=(device_type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)

            if verbose and i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader.dataset)
        val_labels, val_probs, val_loss = run_inference(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_loss)

        if verbose:
            print(f"--- Epoch {epoch + 1} Completed | Train Loss: {train_loss:.4f} "
                  f"| Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} ---")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_auc_history.append(val_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Record the AUC from this same epoch. Reporting max(val_auc_history) instead would
            # describe an epoch whose weights were thrown away.
            best_epoch_auc = val_auc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if 0 < early_stopping_patience <= epochs_no_improve:
            if verbose:
                print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    model.load_state_dict(best_model_weights)

    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'val_auc': val_auc_history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_epoch_auc,
        'max_val_auc': max(val_auc_history),
        'epochs_run': len(train_loss_history),
    }
    return model, history


def summarize_cv(histories: list[dict]) -> dict:
    """
    Aggregate per-fold histories into mean +/- std, the number cross-validation exists to produce.

    :param histories: The ``history`` dicts returned by :func:`train_fold`, one per fold.
    :type histories: list[dict]
    :return: A dict of the aggregated metrics.
    :rtype: dict
    """

    losses = np.array([h['best_val_loss'] for h in histories])
    aucs = np.array([h['best_val_auc'] for h in histories])

    print(f"\n{'Fold':<6}{'Best epoch':>12}{'Val loss':>12}{'Val AUC':>10}")
    for i, h in enumerate(histories):
        print(f"{i + 1:<6}{h['best_epoch'] + 1:>12}{h['best_val_loss']:>12.4f}"
              f"{h['best_val_auc']:>10.4f}")

    print(f"\nVal loss: {losses.mean():.4f} +/- {losses.std():.4f}")
    print(f"Val AUC : {aucs.mean():.4f} +/- {aucs.std():.4f}")

    return {'val_loss_mean': losses.mean(), 'val_loss_std': losses.std(),
            'val_auc_mean': aucs.mean(), 'val_auc_std': aucs.std(),
            'fold_val_losses': losses.tolist(), 'fold_val_aucs': aucs.tolist()}
