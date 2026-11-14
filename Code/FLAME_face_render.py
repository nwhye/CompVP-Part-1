import os
from types import SimpleNamespace

import torch
from flame_pytorch import FLAME

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "Code" / "src"


def create_random_face_dataset(num_samples=100, output_dir=str(BASE_DIR / "generated_objs")):
    os.makedirs(output_dir, exist_ok=True)

    config = SimpleNamespace(
        flame_model_path=str(MODEL_DIR / "flame2023_Open.pkl"),
        static_landmark_embedding_path=str(MODEL_DIR / "flame_static_embedding.pkl"),
        dynamic_landmark_embedding_path=str(MODEL_DIR / "flame_dynamic_embedding.npy"),
        shape_params=300,
        expression_params=100,
        pose_params=6,
        use_face_contour=True,
        use_3D_translation=False,
        optimize_eyeballpose=False,
        optimize_neckpose=False,
        batch_size=1,
    )

    flame_model = FLAME(config)
    flame_model.eval()
    faces = flame_model.faces

    print(f"Generating {num_samples} random 3D face meshes...")

    with torch.no_grad():
        for i in range(num_samples):
            shape_params = torch.randn(1, 300)
            exp_params = torch.randn(1, 100) * 0.5
            pose_params = torch.zeros(1, 6)

            vertices, landmarks = flame_model(
                shape_params=shape_params,
                expression_params=exp_params,
                pose_params=pose_params,
            )

            verts = vertices[0].cpu().numpy()
            write_obj(os.path.join(output_dir, f"face_{i:04d}.obj"), verts, faces)

    print(f"Successfully generated {num_samples} meshes in {output_dir}/")


def write_obj(filepath, verts, faces):
    with open(filepath, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

import trimesh
trimesh.load("generated_objs/face_0021.obj").show()


if __name__ == "__main__":
    create_random_face_dataset(num_samples=100, output_dir="./generated_objs")