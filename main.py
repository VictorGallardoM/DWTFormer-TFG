import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt

from src.data.dataset_loader import load_medmnist_dataset
from src.model.dwtformer import DWTFormer
from src.train.train_model import train
from src.train.evaluate import (
    evaluate_model,
    print_f1_per_class,
    plot_confusion,
    plot_multiclass_roc,
    plot_multilabel_roc_auc_bar,
    plot_classification_heatmap
)

# 🔧 Configuració
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔁 Dataset a entrenar
dataset = "chestmnist"  # 🔄 pots canviar per "pathmnist" o "organamnist" o "chestmnist"

# 🎯 Num de classes per cada dataset
NUM_CLASSES = {
    "pathmnist": 9,
    "chestmnist": 14,
    "organamnist": 11
}

# 🔍 Pèrdua i mode
if dataset == "chestmnist":
    criterion = nn.BCEWithLogitsLoss()
    multilabel = True
else:
    criterion = nn.CrossEntropyLoss()
    multilabel = False

# 📦 Carregar dades
train_loader, val_loader, test_loader = load_medmnist_dataset(dataset, batch_size=BATCH_SIZE)

# 🧠 Model dinàmic segons dataset
model = DWTFormer(num_classes=NUM_CLASSES[dataset]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"📁 Dataset: {dataset.upper()} | Multilabel: {multilabel} | Loss: {type(criterion).__name__}")

# 🚀 Entrenament
history = train(
    model, train_loader, val_loader, criterion, optimizer,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    save_path=f"model/dwtformer_{dataset}.pt",
    multilabel=multilabel
)

# 🧪 Avaluació
test_loss, y_true, y_pred, y_scores = evaluate_model(
    model, test_loader, criterion, DEVICE, multilabel=multilabel
)

class_names = [str(i) for i in range(NUM_CLASSES[dataset])]

print(f"\n📊 Resultats per {dataset.upper()}")
print(f"Test Loss: {test_loss:.4f}")
print_f1_per_class(y_true, y_pred, class_names=class_names)

# 🗂️ Crear carpeta si no existeix
os.makedirs(f"annexos/metrics/{dataset}", exist_ok=True)

if not multilabel:
    # 📊 Visualitzacions per multiclasse
    fig1 = plot_confusion(y_true, y_pred, class_names=class_names)
    fig1.savefig(f"annexos/metrics/{dataset}/confusion_matrix.png")
    plt.close(fig1)

    fig2 = plot_multiclass_roc(y_true, y_scores, n_classes=NUM_CLASSES[dataset], class_names=class_names)
    fig2.savefig(f"annexos/metrics/{dataset}/roc_curve_multiclass.png")
    plt.close(fig2)

else:
    # 📊 Visualitzacions per multilabel
    fig3 = plot_multilabel_roc_auc_bar(y_true, y_scores, class_names)
    fig3.savefig(f"annexos/metrics/{dataset}/roc_auc_per_class.png")
    plt.close(fig3)

# 🔶 Heatmap de mètriques per a tots els casos
fig4 = plot_classification_heatmap(y_true, y_pred, class_names)
fig4.savefig(f"annexos/metrics/{dataset}/classification_metrics_heatmap.png")
plt.close(fig4)

print(f"✅ Resultats guardats per {dataset}")
