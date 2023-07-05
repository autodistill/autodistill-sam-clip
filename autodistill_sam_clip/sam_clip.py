import os
from dataclasses import dataclass

from PIL import Image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity

torch.use_deterministic_algorithms(False)

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from segment_anything import SamAutomaticMaskGenerator

from .helpers import combine_detections, load_SAM

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SAMCLIP(DetectionBaseModel):
    ontology: CaptionOntology
    sam_predictor: SamAutomaticMaskGenerator
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        image_bgr = cv2.imread(input)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # SAM Predictions
        sam_result = self.sam_predictor.generate(image_rgb)

        # mask_annotator = sv.MaskAnnotator()

        # annotated_image = mask_annotator.annotate(
        #     scene=image_bgr, detections=sv.Detections.from_sam(sam_result)
        # )

        # cv2.imshow("image", annotated_image)
        # cv2.waitKey(0)

        valid_detections = []

        labels = self.ontology.prompts()

        nms_data = []

        if len(sam_result) == 0:
            return sv.Detections.empty()

        labels.append("background")

        for mask in sam_result:
            mask_item = mask["segmentation"]

            image = image_rgb.copy()

            # image[mask_item == 0] = 0

            # transform box from xywh to xyxy
            mask["bbox"] = [
                mask["bbox"][0],
                mask["bbox"][1],
                mask["bbox"][0] + mask["bbox"][2],
                mask["bbox"][1] + mask["bbox"][3],
            ]

            # cut out bbox
            image = image[
                int(mask["bbox"][1]) : int(mask["bbox"][3]),
                int(mask["bbox"][0]) : int(mask["bbox"][2]),
            ]

            # fix ValueError: tile cannot extend outside image
            if image.shape[0] == 0 or image.shape[1] == 0:
                continue

            # show me the bbox
            image = Image.fromarray(image)

            image = self.clip_preprocess(image).unsqueeze(0).to(DEVICE)

            cosime_sims = []

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image).to(DEVICE)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text = self.tokenize(labels).to(DEVICE)
                text_features = self.clip_model.encode_text(text).to(DEVICE)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # get cosine similarity between image and text features

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                print(similarity)

                cosime_sims.append(similarity[0][0].item())

            max_prob = None
            max_idx = None

            values, indices = torch.topk(torch.tensor(cosime_sims), 1)

            max_prob = values[0].item()
            max_idx = indices[0].item()

            if max_prob > confidence:
                valid_detections.append(
                    sv.Detections(
                        xyxy=np.array([mask["bbox"]]),
                        confidence=np.array([max_prob]),
                        mask=np.array([mask_item]),
                        class_id=np.array([max_idx]),
                    )
                )
                # (x_min, y_min, x_max, y_max, score, class)
                nms_data.append(
                    (
                        mask["bbox"][0],
                        mask["bbox"][1],
                        mask["bbox"][2],
                        mask["bbox"][3],
                        max_prob,
                        max_idx,
                    )
                )

        final_detections = valid_detections

        nms = sv.non_max_suppression(np.array(nms_data), 0.5)

        final_detections = []

        # nms is list of bools
        for idx, is_valid in enumerate(nms):
            if is_valid:
                final_detections.append(valid_detections[idx])

        return combine_detections(
            final_detections, overwrite_class_ids=[0] * len(final_detections)
        )