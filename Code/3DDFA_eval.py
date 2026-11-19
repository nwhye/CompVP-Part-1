import os, sys, csv
import cv2, yaml, numpy as np, torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "3DDFA_V2")
IN_DIR = os.path.join(BASE, "out", "original")
OUT_DIR = os.path.join(BASE, "out", "model_outputs", "3ddfa" "normal")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, REPO)
os.chdir(REPO)

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
from utils.depth import depth as render_depth
from utils.serialization import ser_to_obj

cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=False, **cfg)
face_boxes = FaceBoxes()

files = sorted(f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
log = []

for file in files:
    stem = os.path.splitext(file)[0]
    img = cv2.imread(os.path.join(IN_DIR, file))   # BGR
    if img is None:
        print(f"{stem}: unreadable, skipped")
        continue
    h, w = img.shape[:2]

    boxes = face_boxes(img)
    detected = len(boxes) > 0
    if not detected:
        boxes = [[0, 0, w - 1, h - 1, 1.0]]

    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_dense = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
    ver_sparse = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    _, pose = calc_pose(param_lst[0])              # yaw, pitch, roll

    np.savez(os.path.join(OUT_DIR, f"{stem}.npz"),
             params=param_lst[0], landmarks=ver_sparse[0],
             vertices=ver_dense[0], pose=np.array(pose), roi_box=roi_box_lst[0])
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_depth.png"),
                render_depth(img, ver_dense, tddfa.tri, show_flag=False))
    ser_to_obj(img, [ver_dense[0]], tddfa.tri, height=h, wfp=os.path.join(OUT_DIR, f"{stem}.obj"))

    log.append([stem, int(detected), *np.round(pose, 2)])
    print(f"{stem}: detected={detected} yaw/pitch/roll={np.round(pose, 1)}")

with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["image", "detected", "yaw", "pitch", "roll"])
    wr.writerows(log)