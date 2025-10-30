# train_triplet_binary.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
from time import gmtime, strftime

from dataset import get_isic2020_data, get_isic2020_data_loaders
from modules import BinaryClassifier, TripletNetwork, set_seed, get_config


###############################################################################
# Helper functions
def triplet_loss(emb_a, emb_p, emb_n, margin=1.0):
    """标准 Triplet Loss"""
    dist_pos = F.pairwise_distance(emb_a, emb_p)
    dist_neg = F.pairwise_distance(emb_a, emb_n)
    loss = F.relu(dist_pos - dist_neg + margin)
    return loss.mean()


def plot_triplet_loss(train_losses):
    """绘制 Triplet 阶段训练损失"""
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Triplet Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('triplet_train_loss.png')
    plt.close()


def plot_binary_training(train_loss, val_loss, train_acc, val_acc):
    """绘制 Binary Classifier 的训练/验证曲线"""
    epochs = len(train_loss)
    plt.figure(figsize=(10, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_loss, label='Train Loss')
    plt.plot(range(1, epochs+1), val_loss, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Binary Classifier Loss')

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_acc, label='Train Acc')
    plt.plot(range(1, epochs+1), val_acc, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Binary Classifier Accuracy')

    plt.tight_layout()
    plt.savefig('binary_train_val_curves.png')
    plt.close()


###############################################################################
# Step 1: Train Triplet Network
def train_triplet(train_loader, model, optimizer, scheduler, epochs, device):
    best_loss = float('inf')
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        losses = []
        for anchor, positive, negative in train_loader:
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

    # 绘制 triplet loss
    plot_triplet_loss(epoch_losses)
    return epoch_losses


###############################################################################
# Step 2b: Binary classifier training
def train_binary_classifier(train_loader, val_loader, triplet_model, classifier, optimizer, scheduler, epochs, device):
    for param in triplet_model.parameters():
        param.requires_grad = False  # freeze encoder

    best_val_auc = -1
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        classifier.train()
        train_losses, train_preds, train_labels = [], [], []

        for img, label in train_loader:
            img, label = img.to(device).float(), label.to(device).float()
            with torch.no_grad():
                emb = triplet_model.encode(img)
            output = classifier(emb).squeeze()
            loss = nn.BCELoss()(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.append(output.detach().cpu())
            train_labels.append(label.cpu())

        avg_train_loss = np.mean(train_losses)
        train_preds = torch.cat(train_preds).numpy()
        train_labels = torch.cat(train_labels).numpy()
        train_acc = accuracy_score(train_labels, train_preds > 0.5)
        train_auc = roc_auc_score(train_labels, train_preds)

        # Validation
        classifier.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(device).float(), label.to(device).float()
                emb = triplet_model.encode(img)
                output = classifier(emb).squeeze()
                val_losses.append(nn.BCELoss()(output, label).item())
                val_preds.append(output.cpu())
                val_labels.append(label.cpu())

        avg_val_loss = np.mean(val_losses)
        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_acc = accuracy_score(val_labels, val_preds > 0.5)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            val_auc = float('nan')

        # 保存历史
        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        print(f"[{strftime('%H:%M:%S', gmtime())}] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(classifier.state_dict(), "binary_classifier_model.pt")
            print(f"Saved best classifier (AUC={best_val_auc:.4f})")

    # 绘制二分类结果
    plot_binary_training(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)


###############################################################################
# Main
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # === 数据加载 ===
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

    # === Step 1: 训练 Triplet ===
    triplet_model = TripletNetwork(emb_dim=config.get('embedding_dims', 128)).to(device)
    optimizer = optim.Adam(triplet_model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_triplet(train_loader, triplet_model, optimizer, scheduler, config['epochs'], device)
    triplet_model.load_state_dict(torch.load("triplet_model.pt"))
    print("Loaded best Triplet model.")

    # === Step 2: Binary classifier ===
    classifier = BinaryClassifier(emb_dim=config.get('embedding_dims', 128)).to(device)
    optimizer_cls = optim.Adam(classifier.parameters(), lr=config['learning_rate'])
    scheduler_cls = ReduceLROnPlateau(optimizer_cls, mode='min', factor=0.5, patience=5)

    train_binary_classifier(train_loader, val_loader, triplet_model, classifier, optimizer_cls, scheduler_cls, config['epochs'], device)


if __name__ == "__main__":
    main()
