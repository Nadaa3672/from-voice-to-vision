"""
Classical machine-learning baselines on the MFCC summary vectors (X_vec).

These give us a reference accuracy before the deep models, and a fair point of
comparison for the report (ML classico vs Deep Learning).
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config


def _split_vec(data: dict):
    s = data["split"]
    out = {}
    for name in ["train", "val", "test"]:
        m = s == name
        out[name] = (data["X_vec"][m], data["y"][m])
    return out


def get_models(seed: int = config.SEED) -> dict:
    """The classical models we compare."""
    return {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "Logistic Reg.": LogisticRegression(max_iter=2000, random_state=seed),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    }


def run_baselines(data: dict, verbose: bool = True):
    """
    Fit each classical model on the train split (features standardized with train stats)
    and evaluate on val and test. Returns (results, scaler).
    """
    parts = _split_vec(data)
    Xtr, ytr = parts["train"]
    Xva, yva = parts["val"]
    Xte, yte = parts["test"]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    results = {}
    for name, model in get_models().items():
        model.fit(Xtr_s, ytr)
        va_acc = accuracy_score(yva, model.predict(Xva_s))
        yte_pred = model.predict(Xte_s)
        te_acc = accuracy_score(yte, yte_pred)
        results[name] = {
            "val_acc": va_acc,
            "test_acc": te_acc,
            "cm": confusion_matrix(yte, yte_pred),
            "report": classification_report(yte, yte_pred,
                                            target_names=config.EMOTIONS, zero_division=0),
            "model": model,
        }
        if verbose:
            print(f"{name:16s}  val={va_acc:.3f}  test={te_acc:.3f}")
    return results, scaler
