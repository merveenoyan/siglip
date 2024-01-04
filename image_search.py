import torch
import gradio as gr
from PIL import Image
from transformers import AutoProcessor, SiglipModel
import faiss
import numpy as np
from huggingface_hub import hf_hub_download
from datasets import load_dataset

hf_hub_download("merve/siglip-faiss-wikiart", "siglip_new.index", local_dir="./")
index = faiss.read_index("./siglip_new.index")

dataset = load_dataset("huggan/wikiart")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dataset = dataset.with_format("torch", device=device)

processor = AutoProcessor.from_pretrained("nielsr/siglip-base-patch16-224")
model = SiglipModel.from_pretrained("nielsr/siglip-base-patch16-224").to(device)


def extract_features_siglip(image):
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
    return image_features

def infer(input_image):
  input_features = extract_features_siglip(input_image)
  input_features = input_features.detach().cpu().numpy()
  input_features = np.float32(input_features)
  faiss.normalize_L2(input_features)
  distances, indices = index.search(input_features, 9)
  gallery_output = []
  for i,v in enumerate(indices[0]):
    sim = -distances[0][i]
    img_resized = dataset["train"][int(v)]['image']
    gallery_output.append(img_resized)
  return gallery_output

gr.Interface(infer, "sketchpad", "gallery",  title="Draw to Search Art 🖼️").launch()
