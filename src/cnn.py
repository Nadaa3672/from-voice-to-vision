"""
CNN on log-Mel spectrograms for Speech Emotion Recognition (PyTorch).

The architecture and the training routine are parameterised by a small
hyper-parameter dict `hp`, so the SAME code is reused in Phase 4 where PSO / FSO /
GA search over these hyper-parameters:

    hp = {
        "lr": 1e-3, "dropout": 0.3, "weight_decay": 1e-4,
        "batch_size": 32, "width": 32,
    }
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src import config


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed: int = config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset (standardizes each spectrogram with TRAIN statistics)
# ----------------------------------------------------------------------
class MelDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = ((X - mean) / (std + 1e-6)).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).unsqueeze(0)  # (1, N_MELS, T)
        return x, int(self.y[i])


def norm_stats(X_train):
    return float(X_train.mean()), float(X_train.std())


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 8, dropout: float = 0.3, width: int = 32):
        super().__init__()
        w = int(width)

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(1, w),
            block(w, w * 2),
            block(w * 2, w * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 4, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------
def _loaders(splits, hp, mean, std):
    bs = int(hp.get("batch_size", 32))
    tr = MelDataset(splits["train"]["X_mel"], splits["train"]["y"], mean, std)
    va = MelDataset(splits["val"]["X_mel"], splits["val"]["y"], mean, std)
    te = MelDataset(splits["test"]["X_mel"], splits["test"]["y"], mean, std)
    return (DataLoader(tr, batch_size=bs, shuffle=True),
            DataLoader(va, batch_size=bs),
            DataLoader(te, batch_size=bs))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for x, y in loader:
        out = model(x.to(device))
        preds.append(out.argmax(1).cpu().numpy())
        tgts.append(y.numpy())
    preds, tgts = np.concatenate(preds), np.concatenate(tgts)
    return accuracy_score(tgts, preds), preds, tgts


def train_cnn(splits, hp, epochs: int = 40, patience: int = 8,
              device=None, verbose: bool = True, seed: int = config.SEED):
    """
    Train the CNN with early stopping on validation accuracy.
    Returns a dict with the trained model, history, and best val / test accuracy.
    """
    set_seed(seed)
    device = device or get_device()
    mean, std = norm_stats(splits["train"]["X_mel"])
    tr_loader, va_loader, te_loader = _loaders(splits, hp, mean, std)

    model = EmotionCNN(n_classes=len(config.EMOTIONS),
                       dropout=float(hp.get("dropout", 0.3)),
                       width=int(hp.get("width", 32))).to(device)

    # class weights handle the neutral imbalance
    classes = np.arange(len(config.EMOTIONS))
    cw = compute_class_weight("balanced", classes=classes, y=splits["train"]["y"])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(hp.get("lr", 1e-3)),
                                 weight_decay=float(hp.get("weight_decay", 1e-4)))

    history = {"train_loss": [], "val_acc": []}
    best_val, best_state, wait = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(tr_loader.dataset)
        val_acc, _, _ = evaluate(model, va_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val, best_state, wait = val_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if verbose:
            print(f"epoch {epoch+1:02d}/{epochs}  loss={train_loss:.3f}  val_acc={val_acc:.3f}")
        if wait >= patience:
            if verbose:
                print(f"  early stopping (no val improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, preds, tgts = evaluate(model, te_loader, device)

    return {
        "model": model, "history": history,
        "best_val_acc": best_val, "test_acc": test_acc,
        "cm": confusion_matrix(tgts, preds),
        "report": classification_report(tgts, preds, target_names=config.EMOTIONS, zero_division=0),
        "norm": (mean, std),
    }
