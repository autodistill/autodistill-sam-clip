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

This repository contains the code supporting the SAM-CLIP base model for use with [Autodistill](https://github.com/autodistill/autodistill).

SAM-CLIP uses the [Segment Anything Model](https://github.com/facebookresearch/segment-anything) to identify objects in an image and assign labels to each image. Then, CLIP is used to find masks that are related to the given prompt.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [SAM-CLIP Autodistill documentation](https://autodistill.github.io/autodistill/base_models/sam-clip/).

## Installation

To use the Grounded SAM base model, you will need to install the following dependency:

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