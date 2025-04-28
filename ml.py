import os
from PIL import Image
from itertools import product
import torch
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict(all_images, user_prompt):
    
    prompt = f'Give me an outfit based on following prompt from the user: {user_prompt}'

    best_combination = None
    highest_score = -1

    for cloth in all_images:
        inputs = processor(text=prompt, images=cloth, return_tensors='pt').to(device)
        output = model(**inputs)

        score = output.logits_per_image.item()

        if score > highest_score:
            highest_score = score
            best_combination = [cloth]


    return best_combination

