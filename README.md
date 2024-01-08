# SigLIP Projects 📎📓

[Taken from the Model Card]

SigLIP is [CLIP](https://huggingface.co/docs/transformers/model_doc/clip), a multimodal model, with a better loss function. The sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. This allows further scaling up the batch size, while also performing better at smaller batch sizes.

A TL;DR of SigLIP by one of the authors can be found [here](https://twitter.com/giffmana/status/1692641733459267713).

## What is this repository for? 👀

This repository shows how you can utilize [SigLIP](https://arxiv.org/abs/2303.15343) for search in different modalities.

📚 It contains:
- A notebook on how to create an embedding index using SigLIP with Hugging Face Transformers and FAISS,
- An image similarity search application that uses the created index,
- An application that compares SigLIP and CLIP.
- An application that compares SigLIP against NLLB-CLIP and CLIP-ViT for multilingual inference.

## Intended uses & limitations

You can use the raw model for tasks like zero-shot image classification and image-text retrieval. See the [SigLIP checkpoints on Hugging Face Hub](https://huggingface.co/models?search=google/siglip) to look for other versions on a task that interests you.

### How to use with 🤗transformers

Here is how to use this model to perform zero-shot image classification:

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-base-patch16-256-i18n")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-i18n")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of 2 cats", "a photo of 2 dogs"]
inputs = processor(text=texts, images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
```

Alternatively, one can leverage the pipeline API which abstracts away the complexity for the user:

```
from transformers import pipeline
from PIL import Image
import requests

# load pipe
image_classifier = pipeline(task="zero-shot-image-classification", model="google/siglip-base-patch16-256-i18n")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
outputs = image_classifier(image, candidate_labels=["2 cats", "a plane", "a remote"])
outputs = [{"score": round(output["score"], 4), "label": output["label"] } for output in outputs]
print(outputs)
```
For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/siglip.html#).



**Citation**

```bibtex
@misc{zhai2023sigmoid,
      title={Sigmoid Loss for Language Image Pre-Training}, 
      author={Xiaohua Zhai and Basil Mustafa and Alexander Kolesnikov and Lucas Beyer},
      year={2023},
      eprint={2303.15343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
