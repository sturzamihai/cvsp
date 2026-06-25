import cv2
import dlib
import torch
import numpy as np
from PIL import Image

from cvsp.models.adv_guard import load_model as load_adv_guard_model
from cvsp.models.lnclip_df import load_model as load_lnclip_model
from cvsp.preprocessing import extract_aligned_face_dlib


class DigitalDefense:
    def __init__(
        self,
        dlib_predictor_path,
        adv_guard_checkpoint_path,
        lnclip_checkpoint_path,
        device="cpu",
    ):
        self.dlib_detector, self.dlib_predictor = self._load_dlib(dlib_predictor_path)

        self.device = device
        self.adv_guard, self.adv_guard_preprocessing = load_adv_guard_model(
            adv_guard_checkpoint_path, device
        )
        self.lnclip, self.lnclip_preprocessing = load_lnclip_model(
            lnclip_checkpoint_path, device
        )

    def _load_dlib(self, predictor_path):
        detector = dlib.get_frontal_face_detector()
        predictor = None

        if predictor_path.exists():
            predictor = dlib.shape_predictor(str(predictor_path))
        else:
            raise RuntimeError("Dlib predictor not found")

        return detector, predictor

    def get_aligned_face(
        self,
        image: np.ndarray | Image.Image,
        scale: float = 1.3,
        use_eye_centers: bool = False,
        input_is_bgr: bool = False,
    ):
        if isinstance(image, Image.Image):
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif input_is_bgr:
            bgr = image
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cropped_bgr, _, _ = extract_aligned_face_dlib(
            self.dlib_detector,
            self.dlib_predictor,
            bgr,
            scale=scale,
            use_eye_centers=use_eye_centers,
        )

        if cropped_bgr is None:
            return None

        return Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

    def __call__(
        self, images, input_is_bgr: bool = False, skip_alignment: bool = False
    ):
        aligned_faces = images if skip_alignment else []

        if not skip_alignment:
            for image in images:
                aligned = self.get_aligned_face(image, input_is_bgr=input_is_bgr)
                if aligned is not None:
                    aligned_faces.append(aligned)

        lnclip_tensors = torch.stack(
            [self.lnclip_preprocessing(img) for img in aligned_faces]
        ).to(self.device)
        adv_guard_tensors = torch.stack(
            [self.adv_guard_preprocessing(img) for img in aligned_faces]
        ).to(self.device)

        with torch.no_grad():
            lnclip_score = self.lnclip(lnclip_tensors).logits_labels.softmax(dim=1)
            adv_guard_score = torch.softmax(self.adv_guard(adv_guard_tensors), dim=1)

            return {
                "deepfake": lnclip_score.mean(dim=0),
                "adversarial": adv_guard_score.mean(dim=0),
            }
