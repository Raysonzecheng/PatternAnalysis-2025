"""
This script evaluates a trained Triplet Network and Binary Classifier on the ISIC 2020 test set.
It computes accuracy, AUC, sensitivity, specificity, and generates:
- Normalized confusion matrix
- ROC curve
- t-SNE embedding visualization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score
)
from sklearn.manifold import TSNE

from modules import BinaryClassifier,TripletNetwork, get_config, set_seed
from dataset import get_isic2020_data, get_isic2020_data_loaders

###############################################################################
# Evaluation Metrics
###############################################################################
def evaluate_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}, "
          f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
    return acc, auc


###############################################################################
# Visualization
###############################################################################
def plot_results(y_true, y_pred, y_prob, embeddings):
    """
    Generate and save:
    - Normalized confusion matrix
    - ROC curve
    - t-SNE embedding visualization
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, None]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("predict_confusion_matrix.png")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_prob):.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("predict_roc_curve.png")
    plt.close()

    # 3. t-SNE Embedding
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, cmap="coolwarm", s=12)
    plt.title("t-SNE Embedding Visualization")
    plt.tight_layout()
    plt.savefig("predict_tsne.png")
    plt.close()


###############################################################################
# Prediction Function
###############################################################################
def predict_models(triplet_model, classifier, loader, device):
    """
    Run inference using Triplet encoder + Binary classifier.
    Returns predictions, probabilities, true labels, and embeddings.
    """
    triplet_model.eval()
    classifier.eval()
    all_probs, all_preds, all_labels, all_emb = [], [], [], []

    with torch.no_grad():
        for anchor, _, _, label in loader:
            anchor = anchor.to(device).float()
            label = label.to(device).long() 

            emb = triplet_model.forward_once(anchor) 
            logits = classifier(emb)                 
            prob = torch.softmax(logits, dim=1)[:, 1] 
            preds = (prob > 0.5).float()

            all_probs.extend(prob.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_emb.extend(emb.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
        np.array(all_emb)
    )

###############################################################################
# Main Entry Point
###############################################################################

def main():
    """
    Load models and test data, run predictions, evaluate, and visualize results.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # Load pretrained models
    triplet_model = TripletNetwork(emb_dim=config['embedding_dims']).to(device)
    classifier = BinaryClassifier(emb_dim=config['embedding_dims']).to(device)
    triplet_model.load_state_dict(torch.load("triplet_model.pt", map_location=device))
    classifier.load_state_dict(torch.load("binary_classifier_model.pt", map_location=device))
    print("Loaded pretrained Triplet and Binary Classifier models.")

    # Load test data
    images, labels = get_isic2020_data(
        config['metadata_path'], config['image_dir'], config.get('data_subset', None)
    )
    _, _, test_loader = get_isic2020_data_loaders(images, labels, train_bs=config['batch_size'])

    # Run predictions
    y_pred, y_prob, y_true, embeddings = predict_models(triplet_model, classifier, test_loader, device)

    # Evaluate and visualize
    acc, auc = evaluate_metrics(y_true, y_pred, y_prob)
    plot_results(y_true, y_pred, y_prob, embeddings)

    print(f"Final Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
