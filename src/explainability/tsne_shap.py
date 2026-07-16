"""
Embedding visualization (t-SNE) and feature attribution (SHAP).

  * extract_embeddings: CNN penultimate features (after global pooling)
  * run_tsne: 2-D projection to inspect class separation
  * feature_names / shap_random_forest: SHAP importances on the MFCC-vector model
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.models.cnn import MelDataset, norm_stats, get_device


def extract_embeddings(model, X_mel, y, mean, std, deltas=True, batch_size=32, device=None):
    """Penultimate CNN features (output of model.features, flattened)."""
    device = device or get_device()
    ds = MelDataset(X_mel, y, mean, std, deltas=deltas, augment=False)
    loader = DataLoader(ds, batch_size=batch_size)
    embs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            f = model.features(x.to(device))     # (B, w*4, 1, 1)
            embs.append(f.flatten(1).cpu().numpy())
    return np.concatenate(embs)


def run_tsne(embeddings, seed=config.SEED, perplexity=30):
    from sklearn.manifold import TSNE
    n = len(embeddings)
    perplexity = min(perplexity, max(5, n // 4))
    return TSNE(n_components=2, init="pca", perplexity=perplexity,
                random_state=seed).fit_transform(embeddings)


def feature_names():
    """Names for the 160-dim MFCC summary vector (mean/std of MFCC and its delta)."""
    n = config.N_MFCC
    return ([f"mfcc_mean_{i}" for i in range(n)] +
            [f"mfcc_std_{i}" for i in range(n)] +
            [f"dmfcc_mean_{i}" for i in range(n)] +
            [f"dmfcc_std_{i}" for i in range(n)])


def shap_random_forest(data, n_estimators=400, seed=config.SEED):
    """
    Fit a Random Forest on the MFCC vectors and compute SHAP values on the test split.
    Returns (shap_values, X_test, feature_names, model).
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    s = data["split"]
    Xtr, ytr = data["X_vec"][s == "train"], data["y"][s == "train"]
    Xte, yte = data["X_vec"][s == "test"], data["y"][s == "test"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    rf.fit(Xtr_s, ytr)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xte_s)
    return shap_values, Xte_s, feature_names(), rf
