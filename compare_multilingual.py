from transformers import AutoTokenizer, CLIPProcessor, SiglipModel, AutoProcessor
import requests
from PIL import Image
from modeling_nllb_clip import NLLBCLIPModel
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch
import numpy as np
import gradio as gr

## NLLB Inference
nllb_clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
nllb_clip_processor = nllb_clip_processor.image_processor
nllb_clip_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

def nllb_clip_inference(image,labels):
  labels = labels.split(",")
  image_inputs = nllb_clip_processor(images=image, return_tensors="pt")
  text_inputs = nllb_clip_tokenizer(labels, padding="longest", return_tensors="pt",)
  nllb_clip_model = NLLBCLIPModel.from_pretrained("visheratin/nllb-clip-base")

  outputs = nllb_clip_model(input_ids = text_inputs.input_ids, attention_mask = text_inputs.attention_mask, pixel_values=image_inputs.pixel_values)
  normalized_tensor = F.softmax(outputs["logits_per_text"], dim=0)
  normalized_tensor = normalized_tensor.detach().numpy()
  return {labels[i]: float(np.array(normalized_tensor)[i]) for i in range(len(labels))}

# SentenceTransformers CLIP-ViT-B-32 
img_model = SentenceTransformer('clip-ViT-B-32')
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

def infer_st(image, texts):
  texts = texts.split(",")
  img_embeddings = img_model.encode(image)
  text_embeddings = text_model.encode(texts)
  cos_sim = util.cos_sim(text_embeddings, img_embeddings)
  return {texts[i]: float(np.array(cos_sim)[i]) for i in range(len(texts))}

### SigLIP Inference

siglip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")


def postprocess_siglip(output, labels):
  return {labels[i]: float(np.array(output[0])[i]) for i in range(len(labels))}

def siglip_detector(image, texts):
  inputs = siglip_processor(text=texts, images=image, return_tensors="pt",
                     padding="max_length")

  with torch.no_grad():
    outputs = siglip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image)
    probs = normalize_tensor(probs)
    
  return probs


def normalize_tensor(tensor):
    # no other normalization works well for visual purposes
    sum_tensor = torch.sum(tensor)
    normalized_tensor = tensor / sum_tensor

    return normalized_tensor

def infer_siglip(image, candidate_labels):
  candidate_labels = [label.lstrip(" ") for label in candidate_labels.split(",")]
  siglip_out = siglip_detector(image, candidate_labels)
  return postprocess_siglip(siglip_out, labels=candidate_labels)

def infer(image, labels):
  st_out = infer_st(image, labels)
  nllb_out = nllb_clip_inference(image, labels)
  siglip_out = infer_siglip(image, labels)
  return st_out, siglip_out, nllb_out


with gr.Blocks() as demo:
  gr.Markdown("# Compare Multilingual Zero-shot Image Classification")
  gr.Markdown("Compare the performance of SigLIP and othe rmodels on zero-shot classification in this Space 👇")
  with gr.Row():
    with gr.Column():
        image_input = gr.Image(type="pil")
        text_input = gr.Textbox(label="Input a list of labels")
        run_button = gr.Button("Run", visible=True)

    with gr.Column():
      st_output = gr.Label(label = "CLIP-ViT Multilingual Output", num_top_classes=3)
      siglip_output = gr.Label(label = "SigLIP Output", num_top_classes=3)
      nllb_output = gr.Label(label = "NLLB-CLIP Output", num_top_classes=3)

  examples = [["./cat.jpg", "eine Katze, köpek, un oiseau"]]
  gr.Examples(
        examples = examples,
        inputs=[image_input, text_input],
        outputs=[st_output,
                 siglip_output,
                 nllb_output],
        fn=infer,
        cache_examples=True
    )
  run_button.click(fn=infer,
                    inputs=[image_input, text_input],
                    outputs=[st_output,
                             siglip_output,
                             nllb_output])

demo.launch()
