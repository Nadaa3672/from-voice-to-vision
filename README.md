# 🎙️➡️🎨 From Voice to Vision
## Explainable, Bio-Inspired Speech Emotion Recognition with Generative Art

An end-to-end framework that **recognizes emotion from the human voice**, optimizes the
recognizer with **bio-inspired algorithms**, makes its decisions **explainable**, and finally
turns the predicted emotion into a **generated artwork** via Stable Diffusion.

> *From Engineering to Arts* — literally: **voice → emotion → explanation → art.**

---

## 🔍 Overview
Speech Emotion Recognition (SER) is challenging: emotional cues are subtle, speaker-dependent,
and spread across time and frequency. This project builds a full pipeline that combines several
techniques seen in the course into a single coherent system:

- **Audio features:** log-Mel spectrograms, MFCC (+ deltas), MFCC-mean vectors
- **Models:** classical ML baselines (SVM / Random Forest) + a **CNN** on spectrograms (+ optional lightweight Transformer)
- **Bio-inspired optimization:** hyper-parameter search with **PSO**, **Flock of Starlings (FSO)**, and **Genetic Algorithm (GA)**, compared head-to-head
- **Explainability (XAI):** Grad-CAM on spectrograms, t-SNE of embeddings, SHAP / LIME
- **Generative art:** the predicted emotion (and its probability blend) conditions **Stable Diffusion** to synthesize an image

---

## 📊 Dataset
**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech.
~1440 speech clips, 24 actors, 8 emotions. Free (Zenodo / Kaggle).

Emotions: `neutral, calm, happy, sad, angry, fearful, disgust, surprised`.

Data splits are **speaker-independent** (actors in the test set never appear in training) to
avoid leakage and to measure true generalization.

---

## 🗂️ Repository structure
```
from-voice-to-vision/
├── notebooks/          # Colab-ready notebooks, one per phase
├── src/
│   ├── config.py       # paths, emotion map, global params
│   ├── data_loader.py  # download + parse RAVDESS
│   ├── preprocessing.py
│   ├── features.py
│   ├── models/         # baseline + CNN (+ transformer)
│   ├── optimization/   # pso.py, fso.py, ga.py
│   ├── explainability/ # gradcam.py, tsne_shap.py
│   └── art/            # emotion_to_image.py (Stable Diffusion)
├── results/            # metrics, figures
└── report/             # paper-style report
```

---

## 🚀 Roadmap
0. Setup + data download
1. Preprocessing & speaker-independent split
2. Feature extraction (log-Mel / MFCC)
3. Baseline models
4. Bio-inspired optimization (PSO vs FSO vs GA)
5. Final model (+ Transformer comparison)
6. Explainability
7. Emotion → Stable Diffusion art
8. Paper-style report

---

## 👤 Author
Nada — *Artificial Intelligence: From Engineering to Arts*
