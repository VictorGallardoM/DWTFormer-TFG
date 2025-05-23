import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', save_path="model/dwtformer_model.pt", multilabel=False):
    """
    Entrena el model amb validació per època.

    Args:
        model (nn.Module): model DWTFormer.
        train_loader, val_loader (DataLoader): dataloaders PyTorch.
        criterion: funció de pèrdua.
        optimizer: optimitzador PyTorch.
        num_epochs (int): nombre d'èpoques.
        device (str): 'cuda' o 'cpu'.
        save_path (str): ruta per guardar el model.
        multilabel (bool): si és classificació multilabel (ChestMNIST)

    Returns:
        dict: historial d'entrenament
    """
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images = images.to(device)
            if multilabel:
                labels = labels.to(device).float()
            else:
                labels = labels.view(-1).to(device).long()  # ✅ robust

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device, multilabel)

        print(f"📘 Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model desat a {save_path}")

    return history


def validate(model, dataloader, criterion, device='cpu', multilabel=False):
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            if multilabel:
                labels = labels.to(device).float()
            else:
                labels = labels.view(-1).to(device).long()  # ✅ robust

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_total += loss.item()

            if not multilabel:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

    avg_loss = loss_total / len(dataloader)
    accuracy = 100.0 * correct / total if not multilabel else 0.0
    return avg_loss, accuracy