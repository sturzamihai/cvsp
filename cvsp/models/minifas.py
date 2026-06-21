from deepface import DeepFace


class FaceAntiSpoofing:
    def __call__(self, image) -> tuple[bool, float] | None:
        """Returns (is_spoof, live_score) or None if no matching face found."""
        faces = DeepFace.extract_faces(img_path=image, anti_spoofing=True)

        largest, largest_surface = None, 0.0
        for f in faces:
            fa = f["facial_area"]
            surface = fa["w"] * fa["h"]

            if surface > largest_surface:
                largest, largest_surface = f, surface

        if largest is None:
            return None

        is_real = largest["is_real"]
        score = largest.get("antispoof_score", 0.5)
        live_score = score if is_real else 1.0 - score

        return not is_real, live_score
