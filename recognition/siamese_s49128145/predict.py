import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)# predict_binary_fixed.py
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
# 评估指标
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
# 绘图函数
def plot_results(y_true, y_pred, y_prob, embeddings):
    # 1️⃣ 混淆矩阵
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

    # 2️⃣ ROC 曲线
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

    # 3️⃣ t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_true, cmap="coolwarm", s=12)
    plt.title("t-SNE Embedding Visualization")
    plt.tight_layout()
    plt.savefig("predict_tsne.png")
    plt.close()


###############################################################################
# 预测函数（Triplet Embedding + Binary Classifier）
def predict_models(triplet_model, classifier, loader, device):
    triplet_model.eval()
    classifier.eval()
    all_probs, all_preds, all_labels, all_emb = [], [], [], []

    with torch.no_grad():
        for anchor, _, _, label in loader:
            anchor = anchor.to(device).float()
            label = label.to(device).float()

            emb = triplet_model.forward_single(anchor)  # ✅ Triplet单输入模式
            prob = classifier(emb).squeeze()
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
# 主函数
def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    # 载入模型
    triplet_model = TripletNetwork(emb_dim=config['embedding_dims']).to(device)
    classifier = BinaryClassifier(emb_dim=config['embedding_dims']).to(device)
    triplet_model.load_state_dict(torch.load("triplet_model.pt", map_location=device))
    classifier.load_state_dict(torch.load("binary_classifier_model.pt", map_location=device))
    print("✅ Loaded pretrained Triplet and Binary Classifier models.")

    # 加载数据
    images, labels = get_isic2020_data(
        config['metadata_path'], config['image_dir'], config.get('data_subset', None)
    )
    _, _, test_loader = get_isic2020_data_loaders(images, labels, train_bs=config['batch_size'])

    # 预测
    y_pred, y_prob, y_true, embeddings = predict_models(triplet_model, classifier, test_loader, device)
    acc, auc = evaluate_metrics(y_true, y_pred, y_prob)
    plot_results(y_true, y_pred, y_prob, embeddings)

    print(f"✅ Final Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

from sklearn.manifold import TSNE

# 🔹 自定义模块
from dataset import get_isic2020_data, get_isic2020_data_loaders
from modules import BinaryClassifier, set_seed, get_config


###############################################################################
# 📊 Evaluation Metrics
###############################################################################
def produce_evaluation_metrics(test_y_pred, test_y_probs, test_y_true):
    """计算 Accuracy、AUC、Sensitivity、Specificity"""
    test_accuracy = accuracy_score(test_y_true, test_y_pred)
    test_auc_roc = roc_auc_score(test_y_true, test_y_probs)

    print(f"✅ Testing Accuracy: {test_accuracy:.4f}")
    print(f"✅ Testing AUC ROC: {test_auc_roc:.4f}")

    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    print(f"✅ Sensitivity (Recall): {sensitivity:.3f}")
    print(f"✅ Specificity: {specificity:.3f}")

    return test_accuracy, test_auc_roc, sensitivity, specificity


###############################################################################
# 📈 Visualization
###############################################################################
def produce_evaluation_figures(test_y_pred, test_y_probs, test_y_true, test_embeddings):
    """绘制 ROC、混淆矩阵、t-SNE"""
    os.makedirs("results", exist_ok=True)

    # 🔸 混淆矩阵
    conf_matrix = confusion_matrix(test_y_true, test_y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix_norm,
        annot=True,
        fmt='.2%',
        cmap='YlGnBu',
        xticklabels=['Normal (0)', 'Melanoma (1)'],
        yticklabels=['Normal (0)', 'Melanoma (1)']
    )
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    # 🔸 ROC 曲线
    fpr, tpr, _ = roc_curve(test_y_true, test_y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='tomato', lw=2,
             label='ROC Curve (AUC = %.3f)' % roc_auc_score(test_y_true, test_y_probs))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/roc_curve.png')
    plt.close()

    # 🔸 t-SNE 可视化嵌入
    print("🌀 Running t-SNE (may take 1–2 minutes)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(test_embeddings)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=test_y_true,
        cmap='coolwarm',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.tight_layout()
    plt.savefig('results/tsne_embeddings.png')
    plt.close()


###############################################################################
# 🔮 Prediction
###############################################################################
def predict_classifier(model: BinaryClassifier, data_loader: DataLoader, device: str):
    """在 test_loader 上进行预测"""
    all_y_pred, all_y_prob, all_y_true, all_embeddings = [], [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)

            # 提取特征嵌入
            # 提取 embedding（即 ResNet 特征）
            embeddings = model.feature_extractor(imgs)
            embeddings = embeddings.view(embeddings.size(0), -1)

            # 送入分类层
            outputs = model.fc_layers(embeddings)


            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            all_y_true.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    return (
        np.array(all_y_pred),
        np.array(all_y_prob),
        np.array(all_y_true),
        np.array(all_embeddings),
    )


###############################################################################
# 🧪 Evaluation Pipeline
###############################################################################
def results_classifier(test_loader: DataLoader, model: BinaryClassifier, device: str):
    """计算指标 + 画图"""
    test_y_pred, test_y_probs, test_y_true, test_embeddings = predict_classifier(model, test_loader, device)
    produce_evaluation_metrics(test_y_pred, test_y_probs, test_y_true)
    produce_evaluation_figures(test_y_pred, test_y_probs, test_y_true, test_embeddings)


###############################################################################
# 🚀 Main
###############################################################################
def main():
    """主评估流程"""
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    config = get_config()

    # 加载模型
    model = BinaryClassifier(emb_dim=config.get('embedding_dims', 128)).to(device)
    model.load_state_dict(torch.load("binary_classifier_model.pt", map_location=device))
    print("✅ Binary classifier model loaded successfully!")

    # 加载数据
    images, labels = get_isic2020_data(
        metadata_path=config['metadata_path'],
        image_dir=config['image_dir'],
        data_subset=config.get('data_subset', None)
    )

    _, _, test_loader = get_isic2020_data_loaders(
        images=images,
        labels=labels,
        train_bs=config['batch_size']
    )

    # 执行评估
    results_classifier(test_loader, model, device)


if __name__ == "__main__":
    main()
