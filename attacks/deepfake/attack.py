from pathlib import Path

import cv2
import numpy as np


class FaceSwap:
    def __init__(self, inswapper_path: Path) -> None:
        self._inswapper_path = inswapper_path
        self._fa_app = None
        self._swapper = None
        self._source_face = None

    def _load(self) -> bool:
        if self._fa_app is not None:
            return True
        try:
            import insightface
            from insightface.app import FaceAnalysis

            self._fa_app = FaceAnalysis()
            self._fa_app.prepare(ctx_id=0)
            if self._inswapper_path.exists():
                self._swapper = insightface.model_zoo.get_model(
                    str(self._inswapper_path)
                )
            return self._swapper is not None
        except ImportError:
            return False

    def prepare(self, image_rgb: np.ndarray) -> bool:
        if not self._load():
            return False
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = self._fa_app.get(bgr)
        if not faces:
            return False
        self._source_face = faces[0]
        return True

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.ready:
            return frame_bgr
        faces = self._fa_app.get(frame_bgr)
        if not faces:
            return frame_bgr
        return self._swapper.get(frame_bgr, faces[0], self._source_face, paste_back=True)

    @property
    def ready(self) -> bool:
        return self._fa_app is not None and self._swapper is not None and self._source_face is not None
