"""
CNN on log-Mel spectrograms for Speech Emotion Recognition (PyTorch).

Improved pipeline (Phase 5):
  * 3-channel input: log-Mel + delta + delta-delta (prosody dynamics)
  * on-the-fly SpecAugment (time/frequency masking) on the training set
  * ReduceLROnPlateau learning-rate scheduler
  * robust fitness: mean of the top-3 validation epochs (less noisy for the optimizers)

Architecture and training are parameterised by a hyper-parameter dict `hp`, reused
by the PSO / FSO / GA optimizers:
    hp = {"lr":1e-3, "dropout":0.3, "weight_decay":1e-4, "batch_size":32, "width":32}
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src import config


def set_seed(seed: int = config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Dataset: standardize with train stats, add delta channels, optional SpecAugment
# ----------------------------------------------------------------------
class MelDataset(Dataset):
    def __init__(self, X, y, mean, std, deltas: bool = True, augment: bool = False,
                 n_freq_mask: int = 2, n_time_mask: int = 2,
                 freq_mask: int = 16, time_mask: int = 24, seed: int = config.SEED):
        base = ((X - mean) / (std + 1e-6)).astype(np.float32)     # (N, H, W)
        if deltas:
            d1 = np.gradient(base, axis=2).astype(np.float32)
            d2 = np.gradient(d1, axis=2).astype(np.float32)
            self.X = np.stack([base, d1, d2], axis=1)             # (N, 3, H, W)
        else:
            self.X = base[:, None, :, :]                          # (N, 1, H, W)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.nfm, self.ntm, self.fm, self.tm = n_freq_mask, n_time_mask, freq_mask, time_mask
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def _spec_augment(self, x):
        C, H, W = x.shape
        for _ in range(self.nfm):
            w = self.rng.integers(0, self.fm + 1)
            if w > 0:
                f0 = self.rng.integers(0, max(1, H - w))
                x[:, f0:f0 + w, :] = 0.0
        for _ in range(self.ntm):
            w = self.rng.integers(0, self.tm + 1)
            if w > 0:
                t0 = self.rng.integers(0, max(1, W - w))
                x[:, :, t0:t0 + w] = 0.0
        return x

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x = self._spec_augment(x)
        return torch.from_numpy(x), int(self.y[i])


def norm_stats(X_train):
    return float(X_train.mean()), float(X_train.std())


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class EmotionCNN(nn.Module):
    def __init__(self, n_classes: int = 8, dropout: float = 0.3, width: int = 32,
                 in_channels: int = 3):
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
            block(in_channels, w),
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
def _loaders(splits, hp, mean, std, deltas, augment):
    bs = int(hp.get("batch_size", 32))
    tr = MelDataset(splits["train"]["X_mel"], splits["train"]["y"], mean, std,
                    deltas=deltas, augment=augment)
    va = MelDataset(splits["val"]["X_mel"], splits["val"]["y"], mean, std,
                    deltas=deltas, augment=False)
    te = MelDataset(splits["test"]["X_mel"], splits["test"]["y"], mean, std,
                    deltas=deltas, augment=False)
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


def train_cnn(splits, hp, epochs: int = 50, patience: int = 12,
              deltas: bool = True, augment: bool = True,
              device=None, verbose: bool = True, seed: int = config.SEED):
    """
    Train the CNN with SpecAugment + LR scheduling and early stopping on val accuracy.
    Returns dict with model, history, best/robust val accuracy and test metrics.
    """
    set_seed(seed)
    device = device or get_device()
    mean, std = norm_stats(splits["train"]["X_mel"])
    in_ch = 3 if deltas else 1
    tr_loader, va_loader, te_loader = _loaders(splits, hp, mean, std, deltas, augment)

    model = EmotionCNN(n_classes=len(config.EMOTIONS),
                       dropout=float(hp.get("dropout", 0.3)),
                       width=int(hp.get("width", 32)),
                       in_channels=in_ch).to(device)

    classes = np.arange(len(config.EMOTIONS))
    cw = compute_class_weight("balanced", classes=classes, y=splits["train"]["y"])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(hp.get("lr", 1e-3)),
                                 weight_decay=float(hp.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4)

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
        scheduler.step(val_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose:
            print(f"epoch {epoch+1:02d}/{epochs}  loss={train_loss:.3f}  val_acc={val_acc:.3f}")
        if wait >= patience:
            if verbose:
                print(f"  early stopping (no val improvement for {patience} epochs)")
            break

    # robust fitness = mean of the 3 best validation epochs (less noisy)
    robust_val = float(np.mean(sorted(history["val_acc"])[-3:]))

    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, preds, tgts = evaluate(model, te_loader, device)

    return {
        "model": model, "history": history,
        "best_val_acc": best_val, "robust_val_acc": robust_val, "test_acc": test_acc,
        "cm": confusion_matrix(tgts, preds),
        "report": classification_report(tgts, preds, target_names=config.EMOTIONS, zero_division=0),
        "norm": (mean, std), "in_channels": in_ch,
    }
