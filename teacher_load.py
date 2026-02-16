import torch
from transformers import pipeline

device_id = 0 if torch.cuda.is_available() else -1

teacher = pipeline(
  task = "depth-estimation",
  model = "LiheYoung/depth-anything-small-hf",  # vits ~= small
  device = device_id,
  dtype = torch.float32,
)

