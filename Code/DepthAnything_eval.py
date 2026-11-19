import os
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

IN_DIR = "./out/original"
OUT_DIR = "./out/model_outputs/depth/normal"
device = "cuda" if torch.cuda.is_available() else "cpu"

depth_pipe = pipeline("depth-estimation",
                      model="depth-anything/Depth-Anything-V2-Large-hf",
                      device=device)

os.makedirs(f"{OUT_DIR}/depth", exist_ok=True)
files = sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))

for file in files:
    stem = os.path.splitext(file)[0]
    img = Image.open(os.path.join(IN_DIR, file)).convert("RGB")
    result = depth_pipe(img)
    result["depth"].save(f"{OUT_DIR}/{stem}.png")
    np.save(f"{OUT_DIR}/{stem}.npy",
            result["predicted_depth"].squeeze().cpu().numpy())