from pathlib import Path
from torchvision.datasets import Places365

current_dir = Parh(__file__).resolve().parent

DATA_ROOT = current_dir / "dataset"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

ds = Places365(
  root = DATA_ROOT,
  split = "val",
  small = False,
  download = True,
)
