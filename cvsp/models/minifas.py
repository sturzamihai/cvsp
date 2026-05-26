from deepface import DeepFace


class FaceAntiSpoofing:
    def __call__(self, image, dlib_face) -> tuple[bool, float] | None:
        """Returns (is_spoof, live_score) or None if no matching face found."""
        try:
            faces = DeepFace.extract_faces(
                img_path=image, anti_spoofing=True, enforce_detection=False
            )
        except Exception:
            return None

        dlib_box = (
            dlib_face.left(),
            dlib_face.top(),
            dlib_face.right(),
            dlib_face.bottom(),
        )
        best, best_iou = None, 0.0
        for f in faces:
            fa = f["facial_area"]
            df_box = (fa["x"], fa["y"], fa["x"] + fa["w"], fa["y"] + fa["h"])
            iou = self._iou(dlib_box, df_box)
            if iou > best_iou:
                best_iou, best = iou, f

        if best is None or best_iou < 0.3:
            return None

        is_real = best["is_real"]
        score = best.get("antispoof_score", 0.5)
        live_score = score if is_real else 1.0 - score
        return not is_real, live_score

    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        if inter == 0:
            return 0.0

        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (a1 + a2 - inter)
