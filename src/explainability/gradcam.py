"""
Grad-CAM for the SER CNN.

Highlights the time-frequency regions of the (log-Mel) spectrogram that most drive
the network's decision for a given emotion. We hook the last convolutional block,
weight its activations by the gradients of the target class, and upsample the
resulting map to the spectrogram size.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx=None):
        """x: (1, C, H, W) tensor already on the model's device. Returns (cam[H,W], class_idx)."""
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1).item())
        out[0, class_idx].backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)          # (1, K, 1, 1)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))     # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def last_conv_block(model):
    """The last convolutional block of EmotionCNN (before global pooling)."""
    return model.features[2]
