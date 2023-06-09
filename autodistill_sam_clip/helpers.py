import os
import urllib.request

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_SAM():
    # Check if segment-anything library is already installed

    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")
    SAM_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "segment_anything")
    SAM_CHECKPOINT_PATH = os.path.join(SAM_CACHE_DIR, "sam_vit_h_4b8939.pth")

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    # Create the destination directory if it doesn't exist
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)

    # Download the file if it doesn't exist
    if not os.path.isfile(SAM_CHECKPOINT_PATH):
        urllib.request.urlretrieve(url, SAM_CHECKPOINT_PATH)

    SAM_ENCODER_VERSION = "vit_h"

    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(
        device=DEVICE
    )
    sam_predictor = SamAutomaticMaskGenerator(sam)

    

    return sam_predictor
