import os

import torch
from PIL import Image
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
depth_pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device=device,
)

for i, file in enumerate(os.listdir("./out/mooney_otsu")):
    mooney_img = Image.open(os.path.join("./out/mooney_otsu", file)).convert("RGB")
    result = depth_pipe(mooney_img)
    depth_map = result["depth"]
    depth_tensor = result["predicted_depth"]
    depth_map.save("mooney_depth_otsu.png")