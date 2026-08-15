"""Small, UI-oriented adapter around the project's CBAM inference pipeline."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.efficientnet_baseline import EfficientNetB0Baseline
from models.efficientnet_cbam import create_efficientnet_cbam
from preprocessing.face_detector import FaceDetector


ProgressCallback = Callable[[str, int], None]


class DemoPredictor:
    """Run a bounded, visual analysis suitable for a live demonstration."""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        model_type: str = "cbam",
        threshold: float = 0.5,
        device: str = "cuda",
        max_frames: int = 32,
    ):
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_frames = max_frames
        self.model_type = model_type
        self.face_detector = FaceDetector(min_confidence=0.90, margin_ratio=0.20, output_size=224)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model(Path(model_path), Path(config_path) if config_path else None)
        self.model.eval()

    def _load_model(self, model_path: Path, config_path: Optional[Path]):
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        if self.model_type == "baseline":
            model = EfficientNetB0Baseline(pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state_dict)
            return model.to(self.device)

        variant, dropout = "default", 0.5
        if config_path and config_path.exists():
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            model_config = config.get("model", {})
            variant = model_config.get("cbam_variant", variant)
            dropout = model_config.get("dropout", dropout)

        model = create_efficientnet_cbam(
            pretrained=False,
            cbam_variant=variant,
            dropout=dropout,
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state_dict)
        return model.to(self.device)

    @staticmethod
    def _sample_indices(total: int, count: int) -> List[int]:
        if total <= count:
            return list(range(total))
        return np.linspace(0, total - 1, count, dtype=int).tolist()

    def _extract_video(self, path: str):
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("The selected video could not be opened.")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        indices = self._sample_indices(total, self.max_frames)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append({
                    "frame_index": index,
                    "timestamp": index / fps if fps else 0,
                    "image": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                })
        capture.release()
        if not frames:
            raise ValueError("No readable frames were found in the selected video.")
        return frames, {"fps": fps, "width": width, "height": height, "duration": total / fps if fps else 0}

    def _extract_image(self, path: str):
        image = cv2.imread(path)
        if image is None:
            raise ValueError("The selected image could not be opened.")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return [{"frame_index": 0, "timestamp": 0, "image": rgb}], {
            "fps": 0, "width": image.shape[1], "height": image.shape[0], "duration": 0,
        }

    def run(self, path: str, is_video: bool = True, progress: Optional[ProgressCallback] = None) -> Dict:
        def update(stage: str, value: int):
            if progress:
                progress(stage, value)

        update("Loading input", 5)
        frames, metadata = self._extract_video(path) if is_video else self._extract_image(path)
        update("Extracting frames", 18)

        results = []
        for position, item in enumerate(frames, 1):
            bbox = self.face_detector.detect_best_face(cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR))
            face = None
            confidence = 0.0
            if bbox:
                confidence = bbox["confidence"]
                face = self.face_detector.crop_face(cv2.cvtColor(item["image"], cv2.COLOR_RGB2BGR), bbox)
            item.update({"bbox": bbox, "face": face, "face_confidence": confidence})
            results.append(item)
            update(f"Detecting faces · frame {position} / {len(frames)}", 18 + int(position / len(frames) * 28))

        valid = [item for item in results if item["face"] is not None]
        if not valid:
            raise ValueError("No face was detected in the selected input. Please try another image or video.")

        update("Running EfficientNet-B0 + CBAM", 58)
        tensors = torch.stack([self.transform(item["face"]) for item in valid]).to(self.device)
        with torch.inference_mode():
            scores = torch.sigmoid(self.model(tensors)).flatten().detach().cpu().numpy()
        for item, score in zip(valid, scores):
            item["fake_probability"] = float(score)
            item["real_probability"] = float(1 - score)
            item["prediction"] = "FAKE" if score >= self.threshold else "REAL"
        for item in results:
            if item.get("face") is None:
                item.update({"fake_probability": None, "real_probability": None, "prediction": "NO FACE"})

        update("Aggregating frame predictions", 78)
        fake_probability = float(np.mean([item["fake_probability"] for item in valid]))
        update("Generating Grad-CAM++ visualization", 90)
        update("Analysis complete", 100)
        return {
            "metadata": metadata,
            "frames": results,
            "threshold": self.threshold,
            "fake_probability": fake_probability,
            "real_probability": 1 - fake_probability,
            "prediction": "FAKE" if fake_probability >= self.threshold else "REAL",
            "total_frames": len(results),
            "detected_faces": len(valid),
            "fake_frames": sum(item["prediction"] == "FAKE" for item in valid),
            "real_frames": sum(item["prediction"] == "REAL" for item in valid),
        }
