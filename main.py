import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset_loader import load_medmnist_dataset
from src.model.dwtformer import DWTFormer
from src.train.train_model import train
from src.train.evaluate import evaluate_model, plot_confusion, plot_multiclass_roc, print_f1_per_class
import matplotlib.pyplot as plt

# 🔧 Configuració
BATCH_SIZE = 64
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Dataset
train_loader, val_loader, _ = load_medmnist_dataset('pathmnist', batch_size=BATCH_SIZE)

# 🧠 Model
model = DWTFormer(num_classes=9)

# 🎯 Pèrdua i optimitzador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🚀 Entrenament
history = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS, device=DEVICE)

# 📦 TEST SET
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

_, _, test_loader = load_medmnist_dataset('pathmnist', batch_size=BATCH_SIZE)

test_loss, y_true, y_pred, y_scores = evaluate_model(model, test_loader, criterion, device)

print(f"📉 Test Loss: {test_loss:.4f}")
print_f1_per_class(y_true, y_pred, class_names=[str(i) for i in range(9)])
plot_confusion(y_true, y_pred, class_names=[str(i) for i in range(9)])
plot_multiclass_roc(y_true, y_scores, n_classes=9, class_names=[str(i) for i in range(9)])
import matplotlib.pyplot as plt

# 📊 Guarda la matriu de confusió com PNG
fig1 = plt.figure()
plot_confusion(y_true, y_pred, class_names=[str(i) for i in range(9)])
fig1.savefig("annexos/metrics/confusion_matrix.png")

# 📈 Guarda la corba ROC com PNG
fig2 = plt.figure()
plot_multiclass_roc(y_true, y_scores, n_classes=9, class_names=[str(i) for i in range(9)])
fig2.savefig("annexos/metrics/roc_curve_multiclass.png")
