import cv2
from retinaface import RetinaFace


class FaceDetector:
    def __init__(
        self,
        min_confidence: float = 0.90,
        margin_ratio: float = 0.20,
        output_size: int = 224,
    ):
        self.min_confidence = min_confidence
        self.margin_ratio = margin_ratio
        self.output_size = output_size

    def detect_best_face(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = RetinaFace.detect_faces(frame_rgb)

        if not isinstance(detections, dict):
            return None

        best_face = None
        best_score = -1

        for _, face in detections.items():
            confidence = float(face.get("score", 0.0))

            if confidence < self.min_confidence:
                continue

            facial_area = face.get("facial_area", None)

            if facial_area is None:
                continue

            x1, y1, x2, y2 = map(int, facial_area)

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            if bbox_width <= 0 or bbox_height <= 0:
                continue

            area = bbox_width * bbox_height
            score = confidence * area

            if score > best_score:
                best_score = score
                best_face = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": confidence,
                }

        return best_face

    def crop_face(self, frame_bgr, bbox):
        image_height, image_width = frame_bgr.shape[:2]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        box_width = x2 - x1
        box_height = y2 - y1

        margin_x = int(box_width * self.margin_ratio)
        margin_y = int(box_height * self.margin_ratio)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image_width, x2 + margin_x)
        y2 = min(image_height, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        face_bgr = frame_bgr[y1:y2, x1:x2]

        if face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(
            face_rgb,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )

        return face_rgb