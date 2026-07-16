"""
Emotion -> Art: turn the classifier's prediction into a generated painting.

Each emotion maps to an artistic prompt (palette, mood, visual elements).
If the prediction is mixed (second emotion above a threshold), the two prompts
are BLENDED — the artwork reflects the emotional nuance, not just the argmax.

Generation uses Stable Diffusion via HuggingFace `diffusers` (inference only,
fits comfortably on a Colab T4).
"""
import numpy as np

from src import config

# ----------------------------------------------------------------------
# Emotion -> artistic prompt fragments
# ----------------------------------------------------------------------
EMOTION_PROMPTS = {
    "neutral":   "minimalist abstract painting, soft gray and beige tones, calm geometry, "
                 "quiet negative space, balanced composition",
    "calm":      "serene impressionist seascape at dawn, pastel blue and soft teal, gentle "
                 "brushstrokes, still water, misty horizon, peaceful atmosphere",
    "happy":     "joyful abstract expressionist painting, radiant yellow and orange, dancing "
                 "shapes, sunlight bursts, confetti of warm colors, celebration",
    "sad":       "melancholic painting, deep blue and gray palette, rain on a window, lonely "
                 "figure in the distance, muted light, quiet sorrow",
    "angry":     "violent expressionist abstract painting, crimson red and black, aggressive "
                 "brushstrokes, eruption of fire, jagged shapes, storm of fury",
    "fearful":   "dark surrealist painting, cold desaturated tones, long shadows, fog, "
                 "twisted forest at night, sense of dread, trembling lines",
    "disgust":   "grotesque abstract painting, acid green and murky brown, decaying textures, "
                 "distorted organic forms, unsettling composition",
    "surprised": "explosive pop-art painting, electric violet and bright cyan, radial burst, "
                 "sparks and lightning, wide-eyed wonder, dynamic motion",
}

STYLE_SUFFIX = "masterpiece, highly detailed, dramatic lighting, oil on canvas"


def build_prompt(probs, blend_threshold: float = 0.25):
    """
    probs: array of 8 class probabilities (order = config.EMOTIONS).
    Returns (prompt, description): the main emotion's prompt, blended with the
    runner-up when the prediction is genuinely mixed.
    """
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(probs)[::-1]
    top, second = order[0], order[1]
    e1, e2 = config.EMOTIONS[top], config.EMOTIONS[second]

    if probs[second] >= blend_threshold:
        prompt = (f"{EMOTION_PROMPTS[e1]}, subtly blended with hints of: "
                  f"{EMOTION_PROMPTS[e2]}, {STYLE_SUFFIX}")
        desc = f"{e1} ({probs[top]:.0%}) + {e2} ({probs[second]:.0%})"
    else:
        prompt = f"{EMOTION_PROMPTS[e1]}, {STYLE_SUFFIX}"
        desc = f"{e1} ({probs[top]:.0%})"
    return prompt, desc


# ----------------------------------------------------------------------
# Stable Diffusion pipeline (inference only)
# ----------------------------------------------------------------------
def load_pipeline(model_id: str = "stabilityai/sd-turbo", device: str = "cuda"):
    """
    SD-Turbo: distilled Stable Diffusion, 1-4 denoising steps, very fast on a T4.
    Fallback (uncomment) : 'stable-diffusion-v1-5/stable-diffusion-v1-5' with ~25 steps.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe


def generate(pipe, prompt: str, seed: int = 0, steps: int = 2, size: int = 512):
    """Generate one image (SD-Turbo: guidance_scale must be 0)."""
    import torch
    g = torch.Generator(device="cuda").manual_seed(int(seed))
    img = pipe(prompt=prompt, num_inference_steps=int(steps), guidance_scale=0.0,
               height=size, width=size, generator=g).images[0]
    return img


# ----------------------------------------------------------------------
# Full voice -> art pipeline
# ----------------------------------------------------------------------
def predict_probs_from_wav(path, model, norm, device, deltas: bool = True):
    """Preprocess a wav, extract log-Mel (+deltas), return the 8 emotion probabilities."""
    import torch
    from src import preprocessing, features

    y = preprocessing.preprocess(path)
    mel = features.log_mel(y)
    mean, std = norm
    base = ((mel - mean) / (std + 1e-6)).astype(np.float32)
    if deltas:
        d1 = np.gradient(base, axis=1).astype(np.float32)
        d2 = np.gradient(d1, axis=1).astype(np.float32)
        x = np.stack([base, d1, d2])
    else:
        x = base[None]
    xt = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(xt), dim=1)[0].cpu().numpy()
    return probs
