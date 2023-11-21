<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png?3"
      >
    </a>
  </p>
</div>

# Autodistill SAM-CLIP

> [!IMPORTANT]  
> This model has been replaced with the SAM-CLIP combination implemented with the Autodistill model combination API. This API enables you to combine using a detection and classification model for auto-labeling. See the code snippet below for an example of using SAM-CLIP with the new API.

## New SAM-CLIP API

First, install the GroundedSAM and CLIP Autodistill modules:

```bash
pip install autodistill autodistill-grounded-sam autodistill-clip
```

To use the new API, choose an abstract class to identify (i.e. "logo") with a base detection model (in the case below, Grounding DINO). Then, choose classes that should be used by the classification model (i.e. "McDonalds", "Burger King"):

```python
from autodistill_clip import CLIP
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
import supervision as sv

from autodistill.core.custom_detection_model import CustomDetectionModel
import cv2

classes = ["McDonalds", "Burger King"]


SAMCLIP = CustomDetectionModel(
    detection_model=GroundedSAM(
        CaptionOntology({"logo": "logo"})
    ),
    classification_model=CLIP(
        CaptionOntology({k: k for k in classes})
    )
)

IMAGE = "logo.jpg"

results = SAMCLIP.predict(IMAGE)

image = cv2.imread(IMAGE)

annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{classes[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, _ in results
]

annotated_frame = annotator.annotate(
    scene=image.copy(), detections=results
)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, labels=labels, detections=results
)

sv.plot_image(annotated_frame, size=(8, 8))
```

## Archived Contents

This repository contains the code supporting the SAM-CLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

SAM-CLIP uses the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to identify objects in an image and assign labels to each image. Then, CLIP is used to find masks that are related to the given prompt.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SAM-CLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/sam-clip/).

## Installation

To use the SAM-CLIP base model, you will need to install the following dependency:

```bash
pip3 install autodistill-sam-clip
```

## Quickstart

```python
from autodistill_sam_clip import SAMCLIP
from autodistill_yolov8 import YOLOv8


# define an ontology to map class names to our CLIP prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = SAMCLIP(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!
