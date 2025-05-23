from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def print_f1_per_class(y_true, y_pred, class_names=None):
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("🔎 F1-score per classe:\n")
    print(report)


def plot_confusion(y_true, y_pred, class_names=None, normalize='true'):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Matriu de confusió (normalitzada)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_multiclass_roc(y_true, y_scores, n_classes, class_names=None):
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        name = class_names[i] if class_names else f"Classe {i}"
        plt.plot(fpr[i], tpr[i], label=f'{name} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Corbes ROC per classe')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_multilabel_roc_auc_bar(y_true, y_scores, class_names):
    roc_auc = roc_auc_score(y_true, y_scores, average=None)
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=class_names, y=roc_auc)
    plt.title("🔍 ROC-AUC per classe")
    plt.ylabel("AUC")
    plt.xlabel("Classes")
    plt.tight_layout()
    return fig


def plot_classification_heatmap(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose().iloc[:len(class_names)]
    fig = plt.figure(figsize=(10, 5))
    sns.heatmap(df[['precision', 'recall', 'f1-score']], annot=True, cmap="Blues", fmt=".2f")
    plt.title("📊 F1, Precision i Recall per classe")
    plt.tight_layout()
    return fig


def evaluate_model(model, dataloader, criterion, device='cpu', multilabel=False):
    model.eval()
    model.to(device)

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float() if multilabel else labels.to(device).squeeze().long()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if multilabel:
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true.extend(labels.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(torch.tensor(probs), dim=1).cpu().numpy()
                y_true.extend(labels.cpu().numpy())

            y_pred.extend(preds)
            y_scores.extend(probs)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(y_true), np.array(y_pred), np.array(y_scores)
