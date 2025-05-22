import torch
import pytest
from src.model.dwtformer import DWTFormer
from src.data.dataset_loader import load_medmnist_dataset
from src.train.evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Test 1: Forward pass amb batch variable
@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_model_forward_shape(batch_size):
    model = DWTFormer(num_classes=9)
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 9)

# Test 2: Càrrega del dataset i formes
def test_dataset_loader_shapes():
    train_loader, _, _ = load_medmnist_dataset("pathmnist", batch_size=16)
    images, labels = next(iter(train_loader))
    assert images.shape == (16, 1, 28, 28)
    assert labels.shape[0] == 16

# Test 3: Entrenament mínim
def test_training_step():
    model = DWTFormer(num_classes=9)
    images = torch.randn(2, 1, 28, 28)
    labels = torch.randint(0, 9, (2,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0

# Test 4: Funció d’avaluació
def test_evaluate_model_outputs():
    model = DWTFormer(num_classes=3)
    images = torch.randn(10, 1, 28, 28)
    labels = torch.randint(0, 3, (10,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=5)
    criterion = nn.CrossEntropyLoss()

    loss, y_true, y_pred, y_scores = evaluate_model(model, loader, criterion)
    assert len(y_true) == 10
    assert y_scores.shape == (10, 3)
