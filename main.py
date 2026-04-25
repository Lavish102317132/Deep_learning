import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import os

device = "mps" if torch.backends.mps.is_available() else "cpu"


# ======================
# DATASET
# ======================
class RoadDataset(Dataset):
    def __init__(self, folder, size=128, occlusion=False):
        self.folder = folder
        self.size = size
        self.occlusion = occlusion

        all_files = os.listdir(folder)

        self.files = []
        for f in all_files:
            if "_sat" in f:
                mask_name = f.replace("_sat.jpg", "_mask.png")
                if mask_name in all_files:
                    self.files.append(f)

        print(f"Loaded {len(self.files)} valid samples from {folder}")

    def cutout(self, img):
        h, w, _ = img.shape
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        s = np.random.randint(20, 40)
        img[y:y+s, x:x+s] = 0
        return img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        img_path = os.path.join(self.folder, file)
        mask_path = os.path.join(self.folder, file.replace("_sat.jpg", "_mask.png"))

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, (self.size, self.size))
        img = img / 255.0

        mask = cv2.resize(mask, (self.size, self.size))
        mask = (mask > 127).astype(np.float32)

        if self.occlusion:
            img = self.cutout(img)

        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask).unsqueeze(0)


# ======================
# MODEL (U-NET)
# ======================
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.u1 = DoubleConv(256 + 128, 128)
        self.u2 = DoubleConv(128 + 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        u1 = self.up(d3)
        u1 = self.u1(torch.cat([u1, d2], dim=1))

        u2 = self.up(u1)
        u2 = self.u2(torch.cat([u2, d1], dim=1))

        return torch.sigmoid(self.out(u2))


# ======================
# METRICS
# ======================
def iou(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)


def f1(pred, target):
    pred = (pred > 0.5).float()
    tp = (pred * target).sum()
    precision = tp / (pred.sum() + 1e-6)
    recall = tp / (target.sum() + 1e-6)
    return 2 * precision * recall / (precision + recall + 1e-6)


# ======================
# TRAIN
# ======================
def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(loader):
            print(f"Train Batch {i+1}/{len(loader)} Loss={loss.item():.4f}")

    return total_loss / len(loader)


# ======================
# EVALUATE
# ======================
def evaluate(model, loader):
    model.eval()
    iou_score, f1_score = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            iou_score += iou(pred, y).item()
            f1_score += f1(pred, y).item()

            if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(loader):
                print(f"Val Batch {i+1}/{len(loader)}")

    return iou_score / len(loader), f1_score / len(loader)


# ======================
# MAIN
# ======================
print("Loading dataset...")

full_data = RoadDataset("archive/train", occlusion=True)

train_size = int(0.8 * len(full_data))
val_size = len(full_data) - train_size

train_data, val_data = random_split(full_data, [train_size, val_size])

print("Train size:", len(train_data))
print("Valid size:", len(val_data))

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
val_loader = DataLoader(val_data, batch_size=2)

model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

epochs = 5

print("Starting training...")

for e in range(epochs):
    print(f"\nEpoch {e} started")

    loss = train(model, train_loader, optimizer, loss_fn)
    iou_score, f1_score = evaluate(model, val_loader)

    print(f"\nEpoch {e}: Loss={loss:.4f}, IoU={iou_score:.4f}, F1={f1_score:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved!")
