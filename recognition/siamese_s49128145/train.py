from time import gmtime, strftime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score


from dataset import get_isic2020_data, get_isic2020_data_loaders

from modules import BinaryClassifier, set_seed, get_config


###############################################################################
### Helpers: predict + results
def predict_classifier(model: nn.Module, data_loader: DataLoader, device: str):
   
    model.eval()
    preds = []
    probs = []
    trues = []
    logits_list = []

    with torch.no_grad():
        for batch in data_loader:
            # Expect (images, labels)
            images, labels = batch
            images = images.to(device).float()
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(logits, dim=1)

            preds.extend(pred_classes.cpu().numpy().tolist())
            probs.extend(probabilities[:, 1].cpu().numpy().tolist()) 
            trues.extend(labels.cpu().numpy().tolist())
            logits_list.extend(logits.cpu().numpy().tolist())

    return np.array(preds), np.array(probs), np.array(trues), np.array(logits_list)


def results_classifier(test_loader: DataLoader, model: nn.Module, device: str) -> None:
    """
    Print test set metrics (accuracy + AUC) and return them.
    """
    preds, probs, trues, _ = predict_classifier(model, test_loader, device)
    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except ValueError:
        auc = float('nan')  # in case only one class present in y_true
    print(f"Test Accuracy: {acc:.4f}  |  Test AUC: {auc:.4f}")
    return acc, auc


###############################################################################
### Training loop
def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    epochs: int,
    device: str
) -> None:
    """
    Train a binary classification model and save best model (by validation AUC) to disk.
    """


    train_loss_per_epoch = []
    val_loss_per_epoch = []


    for epoch in range(epochs):
        model.train()
        running_losses = []

        for batch in train_loader:
            images, labels = batch
            images = images.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.detach().cpu().numpy())

        avg_train_loss = float(np.mean(running_losses)) if running_losses else 0.0

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(device).float()
                labels = labels.to(device).long()

                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.detach().cpu().numpy())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        

        # Record
        train_loss_per_epoch.append(avg_train_loss)
        val_loss_per_epoch.append(avg_val_loss)


        # Print progress
        print(
            f"[{strftime('%H:%M:%S', gmtime())}] Epoch {epoch+1:>2}/{epochs} "
        )

 

   


###############################################################################
### Main
def main() -> None:
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config.get('data_subset', None)
    )

    # Ensure get_isic2020_data_loaders returns DataLoader objects for classification
    train_loader, val_loader, test_loader = get_isic2020_data_loaders(
        images=images,
        labels=labels,
        train_bs=config['batch_size']
    )

    model = BinaryClassifier(emb_dim=config.get('embedding_dims', 128)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=config['epochs'],
        device=device
    )

    # Load best model (if saved)
    try:
        model.load_state_dict(torch.load("binary_classifier_model.pt"))
        print("Loaded best model from binary_classifier_model.pt")
    except Exception as e:
        print("Warning: could not load saved best model:", e)

    # Evaluate on test set
    results_classifier(test_loader, model, device)


if __name__ == "__main__":
    main()
