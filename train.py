import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import MultiModalDataset
from multimodal_model import MultiModalNet

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x_tab, x_imgs, y in loader:
        x_tab, x_imgs, y = x_tab.to(device), x_imgs.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x_tab, x_imgs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_tab.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += x_tab.size(0)
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_tab, x_imgs, y in loader:
            x_tab, x_imgs, y = x_tab.to(device), x_imgs.to(device), y.to(device)
            logits = model(x_tab, x_imgs)
            loss = criterion(logits, y)
            total_loss += loss.item() * x_tab.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += x_tab.size(0)
    return total_loss / total, correct / total

if __name__ == "__main__":
    num_features = ['Age', 'Body_Weight_kg', 'Body_Height_cm', 'BMI', 'Gender_enc']
    image_cols = ['u_photo', 'l_photo', 'hp_photo', 'hd_photo', 'hdf_photo']
    label_col = 'Growth_Phase_enc'
    batch_size = 8
    epochs = 10
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data (change fold as needed)
    train_ds = MultiModalDataset('train_fold1.csv', num_features, image_cols, label_col, augment=True)
    val_ds = MultiModalDataset('val_fold1.csv', num_features, image_cols, label_col, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = MultiModalNet(tabular_dim=len(num_features), n_classes=3, cnn_name='resnet18').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d}: Train loss {train_loss:.4f}, acc {train_acc:.3f} | Val loss {val_loss:.4f}, acc {val_acc:.3f}")

    # (Optional) Save trained model
    torch.save(model.state_dict(), "multimodal_fold1.pth")
