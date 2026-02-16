import numpy as np
from pathlib import Path
from teacher_load import teacher
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent

DATA_ROOT = current_dir / "dataset"

IMG_DIR = DATA_ROOT / "val_large"
OUT_DIR = DATA_ROOT / "pseudo_labels"

img_paths = sorted(IMG_DIR.rglob("*.jpg"))
#print("Found images:", len(img_paths))

# path generation

num_saved = 0
num_skipped = 0
num_failed = 0

torch.set_grad_enabled(False)

for p in tqdm(img_paths):
    try:
        out_path = OUT_ROOT / f"{p.stem}.npy"

        # resume 기능
        if out_path.exists():
            num_skipped += 1
            continue

        with Image.open(p) as im:
            img = im.convert("RGB")

        out = teacher(img)
        pred = out["predicted_depth"]

        if isinstance(pred, torch.Tensor):
            depth = pred.squeeze().float().cpu().numpy()
        else:
            depth = np.array(pred, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]

        np.save(out_path, depth.astype(SAVE_DTYPE))
        num_saved += 1

    except Exception as e:
        num_failed += 1
        print(f"[FAIL] {p.name} -> {e}")
