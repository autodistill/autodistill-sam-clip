import os
from dataclasses import dataclass

from PIL import Image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

import cv2
import torch

torch.use_deterministic_algorithms(False)

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from segment_anything import SamAutomaticMaskGenerator

from helpers import load_SAM

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAMCLIP(DetectionBaseModel):
    ontology: CaptionOntology
    sam_predictor: SamAutomaticMaskGenerator

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology
        self.sam_predictor = load_SAM()

        if not os.path.exists(f"{HOME}/.cache/autodistill/clip"):
            os.makedirs(f"{HOME}/.cache/autodistill/clip")

            os.system("pip install ftfy regex tqdm")
            os.system(
                f"cd {HOME}/.cache/autodistill/clip && pip install git+https://github.com/openai/CLIP.git"
            )

        # add clip path to path
        sys.path.insert(0, f"{HOME}/.cache/autodistill/clip/CLIP")

        import clip

        model, preprocess = clip.load("ViT-B/32", device=DEVICE)

        self.clip_model = model
        self.clip_preprocess = preprocess
        self.tokenize = clip.tokenize

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = cv2.imread(input)

        # SAM Predictions
        sam_result = self.sam_predictor.generate(image)
        detections = sv.Detections.from_sam(sam_result=sam_result)

        # CLIP Predictions
        for i, _ in enumerate(detections):
            labels = self.ontology.prompts()

            indices = list(range(len(labels)))

            # image is mask
            mask = detections[i].mask

            # extract mask from image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
            # remove outside mask
            image[mask == False] = 0
            image = Image.fromarray(image)

            image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)
            text = self.tokenize(labels).to(DEVICE)

            with torch.no_grad():
                logits_per_image, _ = self.clip_model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            if probs[0][indices[0]] < confidence:
                detections.mask[i] = None
                detections.confidence[i] = None
                detections.class_id[i] = None

        return detections
