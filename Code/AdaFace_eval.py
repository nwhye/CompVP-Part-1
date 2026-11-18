import os, sys, csv
import numpy as np, torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
ADA = os.path.join(BASE, "AdaFace")
CKPT = os.path.join(ADA, "pretrained", "adaface_ir101_webface12m.ckpt")
ARCH = "ir_101"
IN_DIR = os.path.join(BASE, "out", "mooney_otsu")
OUT_DIR = os.path.join(BASE, "out", "model_outputs", "adaface")
CROP_FRAC = 0.8
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ADA)
import net

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
model = net.build_model(ARCH)
model.load_state_dict({k[6:]: v for k, v in ckpt.items() if k.startswith("model.")})
model.eval()


def to_input(pil_rgb):
    a = (np.array(pil_rgb)[:, :, ::-1] / 255.0 - 0.5) / 0.5
    return torch.from_numpy(a.transpose(2, 0, 1).copy())[None].float()


def fixed_crop(img, size=112, frac=CROP_FRAC):
    w, h = img.size
    s = int(min(w, h) * frac)
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s)).resize((size, size), Image.BILINEAR)


files = sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
embs, names, rows = [], [], []

with torch.no_grad():
    for f in files:
        path = os.path.join(IN_DIR, f)
        crop = fixed_crop(Image.open(path).convert("RGB"))
        feat, norm = model(to_input(crop))
        embs.append(feat[0].numpy())
        names.append(os.path.splitext(f)[0])
        rows.append([names[-1], round(norm.item(), 3)])
        print(f"{names[-1]}: norm={norm.item():.2f}")

np.savez(os.path.join(OUT_DIR, "embeddings.npz"), names=np.array(names), emb=np.stack(embs))
with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
    wr = csv.writer(fh)
    wr.writerow(["image", "feature_norm"])
    wr.writerows(rows)