import os
import numpy as np
import torch
import diffusers
from PIL import Image

IN_DIR = "./out/mooney_otsu"
OUT_DIR = "./out/model_outputs/marigold"
NORMALS_DIR = os.path.join(OUT_DIR, "normals")
ALBEDO_DIR = os.path.join(OUT_DIR, "albedo")
os.makedirs(NORMALS_DIR, exist_ok=True)
os.makedirs(ALBEDO_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
STEPS = 4

normals_pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-v1-1", torch_dtype=dtype).to(device)
iid_pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
    "prs-eth/marigold-iid-appearance-v1-1", torch_dtype=dtype).to(device)

files = sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))

for f in files:
    stem = os.path.splitext(f)[0]
    img = Image.open(os.path.join(IN_DIR, f)).convert("RGB")

    # Normals
    n_out = normals_pipe(img, num_inference_steps=STEPS)
    np.save(os.path.join(NORMALS_DIR, f"{stem}.npy"), n_out.prediction[0])
    normals_pipe.image_processor.visualize_normals(n_out.prediction)[0].save(
        os.path.join(NORMALS_DIR, f"{stem}.png"))

    # Albedo
    i_out = iid_pipe(img, num_inference_steps=STEPS)
    vis = iid_pipe.image_processor.visualize_intrinsics(
        i_out.prediction, iid_pipe.target_properties)[0]
    for name, im in vis.items():
        im.save(os.path.join(ALBEDO_DIR, f"{stem}_{name}.png"))
    np.save(os.path.join(ALBEDO_DIR, f"{stem}_raw.npy"), np.asarray(i_out.prediction))

    print(f"{stem}: done, intrinsic outputs = {list(vis.keys())}")