from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, classification_report
import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def print_f1_per_class(y_true, y_pred, class_names=None):
    """
    Mostra el F1-score per classe de forma clara.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("🔎 F1-score per classe:\n")
    print(report)


def plot_multiclass_roc(y_true, y_scores, n_classes, class_names=None):
    """
    Dibuixa corbes ROC per cada classe en classificació multiclass.
    """
    y_bin = label_binarize(y_true, classes=range(n_classes))  # shape: (N, C)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
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
    plt.show()

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """
    Avalua el model sobre el conjunt de test.

    Args:
        model (nn.Module): model entrenat
        test_loader (DataLoader): test set
        criterion: funció de pèrdua
        device (str): 'cuda' o 'cpu'

    Returns:
        tuple: (loss, y_true, y_pred, y_scores)
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.squeeze().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    return avg_loss, np.array(y_true), np.array(y_pred), np.array(y_scores)


def plot_confusion(y_true, y_pred, class_names=None, normalize='true'):
    """
    Mostra la matriu de confusió amb anotacions clares.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Matriu de confusió (normalitzada)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
