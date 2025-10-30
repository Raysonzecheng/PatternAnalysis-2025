"""
Main training script for Triplet Network and Binary Classifier on ISIC 2020 data.
Steps:
1. Train Triplet Network to learn embeddings
2. Train Binary Classifier using frozen triplet embeddings
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from time import gmtime, strftime

from dataset import get_isic2020_data, get_isic2020_data_loaders
from modules import BinaryClassifier, TripletNetwork, set_seed, get_config

###############################################################################
# Helper Functions
###############################################################################
def triplet_loss(emb_a, emb_p, emb_n, margin=1.0):
    """
    Standard Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
    """
    dist_pos = F.pairwise_distance(emb_a, emb_p)
    dist_neg = F.pairwise_distance(emb_a, emb_n)
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def plot_triplet_loss(train_losses):
    """
    Plot training loss curve for Triplet Network
    """
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Triplet Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('triplet_train_loss.png')
    plt.close()

###############################################################################
# Step 1: Train Triplet Network
###############################################################################
def train_triplet(train_loader, model, optimizer, scheduler, epochs, device):
    """
    Train Triplet Network using triplet loss.
    Saves best model based on lowest training loss.
    """
    best_loss = float('inf')
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for anchor, positive, negative , _ in train_loader:
            anchor, positive, negative = (
                anchor.to(device).float(),
                positive.to(device).float(),
                negative.to(device).float()
            )

            optimizer.zero_grad()
            emb_a, emb_p, emb_n = model(anchor, positive, negative)
            loss = triplet_loss(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)

        print(f"[{strftime('%H:%M:%S', gmtime())}] Epoch {epoch+1:>2}/{epochs} | Train Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "triplet_model.pt")
            print(f"Saved model with lowest loss: {avg_loss:.4f}")

    plot_triplet_loss(epoch_losses)
    return epoch_losses

###############################################################################
# Step 2: Train Binary Classifier
###############################################################################
def train_binary_classifier(train_loader, val_loader, triplet_model, classifier, optimizer, scheduler, epochs, device):
    """
    Train binary classifier using frozen triplet embeddings.
    Evaluates performance and saves best model.
    """
    training_features, training_labels = [], []
    validation_features, validation_labels = [], []

    # Extract embeddings
    with torch.no_grad():
        for anchor, _, _, label in train_loader:
            features = triplet_model.forward_once(anchor.to(device))
            training_features.append(features)
            training_labels.append(label.to(device))
        for anchor, _, _, label in val_loader:
            features = triplet_model.forward_once(anchor.to(device))
            validation_features.append(features)
            validation_labels.append(label.to(device))

     # Freeze triplet model
    for param in triplet_model.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()

    # Training loop
    classifier.train()
    t_loss_total = []
    for features, labels in zip(training_features, training_labels):
        optimizer.zero_grad()
        out = classifier(features)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_loss_total.append(loss.item())

    # Validation loop
    classifier.eval()
    v_loss_total = []
    with torch.no_grad():
        for features, labels in zip(validation_features, validation_labels):
            out = classifier(features)
            loss = criterion(out, labels)
            v_loss_total.append(loss.item())

    # Metrics
    avg_train_loss = np.mean(t_loss_total)
    avg_val_loss = np.mean(v_loss_total)

    train_preds = torch.cat([classifier(f).argmax(1) for f in training_features])
    train_labels = torch.cat(training_labels)
    train_acc = (train_preds == train_labels).float().mean().item()
    train_probs = torch.cat([F.softmax(classifier(f), dim=1)[:,1] for f in training_features]).detach()
    train_auc = roc_auc_score(train_labels.cpu(), train_probs.cpu())

    val_preds = torch.cat([classifier(f).argmax(1) for f in validation_features])
    val_labels = torch.cat(validation_labels)
    val_acc = (val_preds == val_labels).float().mean().item()
    val_probs = torch.cat([F.softmax(classifier(f), dim=1)[:,1] for f in validation_features]).detach()
    val_auc = roc_auc_score(val_labels.cpu(), val_probs.cpu())

    # Scheduler step
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(avg_val_loss)
    else:
        scheduler.step()

    print(f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

    torch.save(classifier.state_dict(), "binary_classifier_model.pt")
    print(f"Saved classifier (AUC={val_auc:.4f})")

###############################################################################
# Main Entry Point
###############################################################################
def main():
    """
    Main training pipeline:
    - Load data
    - Train Triplet Network
    - Train Binary Classifier
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # Load data
    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config.get('data_subset', None)
    )

    train_loader, val_loader, test_loader = get_isic2020_data_loaders(
        images=images,
        labels=labels,
        train_bs=config['batch_size']
    )

    # Step 1: Train Triplet Network
    triplet_model = TripletNetwork(emb_dim=config.get('embedding_dims', 128)).to(device)
    optimizer = optim.Adam(triplet_model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_triplet(train_loader, triplet_model, optimizer, scheduler, config['epochs'], device)
    triplet_model.load_state_dict(torch.load("triplet_model.pt"))
    print("Loaded best Triplet model.")

    # Step 2: Train Binary Classifier
    classifier = BinaryClassifier(emb_dim=config.get('embedding_dims', 128)).to(device)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    scheduler_cls = ReduceLROnPlateau(optimizer_cls, mode='min', factor=0.5, patience=5)

    train_binary_classifier(train_loader, val_loader, triplet_model, classifier, optimizer_cls, scheduler_cls, config['epochs'], device)

if __name__ == "__main__":
    main()
