"""Grad-CAM++ visualization for the EfficientNet-CBAM demo."""

import cv2
import numpy as np
import torch
import tempfile
from pathlib import Path


def gradcam_plus_plus(model, face_rgb: np.ndarray, transform, device) -> np.ndarray:
    """Return a color heatmap blended with the supplied RGB face crop."""
    activations = {}
    gradients = {}
    target_layer = model.features[-1]

    def save_activation(_, __, output):
        activations["value"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(save_activation)
    try:
        tensor = transform(face_rgb).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logit = model(tensor)
        logit[:, 0].backward()
        activation = activations["value"].detach()[0]
        gradient = activations["value"].grad.detach()[0]

        # This is the Grad-CAM++ weighting formulation, reduced over channels.
        grad_2 = gradient.pow(2)
        grad_3 = gradient.pow(3)
        denominator = 2 * grad_2 + (activation * grad_3).sum(dim=(1, 2), keepdim=True)
        alpha = grad_2 / (denominator + 1e-7)
        weights = (alpha * torch.relu(gradient)).sum(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        cam = cam.cpu().numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        cam = cv2.resize(cam, (face_rgb.shape[1], face_rgb.shape[0]))

        heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.55 * face_rgb + 0.45 * heatmap, 0, 255).astype(np.uint8)
        return {"heatmap": heatmap, "overlay": overlay}
    finally:
        handle.remove()


def gradcam_video(model, frames, transform, device, fps: float = 5.0, duration: float = 0, speed: float = 0.4) -> bytes:
    """Create a slow MP4 containing only Grad-CAM++ face crops."""
    face_frames = [frame for frame in frames if frame.get("face") is not None]
    if not face_frames:
        raise ValueError("No analyzed frames are available for Grad-CAM++ video generation.")

    height, width = face_frames[0]["face"].shape[:2]
    # Frames are sampled for the demo, so the source FPS would make the
    # exported sequence play much too quickly. Preserve the sampled timeline.
    timeline_duration = duration or (frames[-1].get("timestamp", 0) - frames[0].get("timestamp", 0))
    sampled_fps = len(face_frames) / max(timeline_duration, 1 / max(float(fps), 1.0))
    output_fps = max(0.1, sampled_fps * speed)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output:
        output_path = Path(output.name)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    try:
        for frame in face_frames:
            visual = gradcam_plus_plus(model, frame["face"], transform, device)
            writer.write(cv2.cvtColor(visual["overlay"], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    try:
        return output_path.read_bytes()
    finally:
        output_path.unlink(missing_ok=True)
