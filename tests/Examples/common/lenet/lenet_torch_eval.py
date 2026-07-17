"""Validates the exported LeNet PyTorch model on the MNIST test dataset."""

import os
import struct
import numpy as np
import torch
from torch import nn

MNIST_PATH = "tests/Examples/common/mnist/data/"
IMAGES_PATH = MNIST_PATH + "t10k-images-idx3-ubyte"
LABELS_PATH = MNIST_PATH + "t10k-labels-idx1-ubyte"

LENET_PATH = "tests/Examples/common/lenet/"
TRACED_MODEL_PATH = LENET_PATH + "traced_model.pt"
MODEL_STATE_PATH = LENET_PATH + "model_state.pt"


class MultilayerPerceptron(nn.Module):
  """Self-contained LeNet-5 architecture so evaluation works anywhere outside google3."""

  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(
        in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
    )
    self.conv2 = nn.Conv2d(
        in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
    )
    self.linear1 = nn.Linear(400, 120)
    self.linear2 = nn.Linear(120, 84)
    self.feature_extractor = nn.Sequential(
        self.conv1,
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        self.conv2,
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
    )

    self.classifier = nn.Sequential(
        self.linear1,  # in_features = 16x5x5
        nn.ReLU(),
        self.linear2,
        nn.ReLU(),
        nn.Linear(84, 10),
    )

  def forward(self, x):
    a1 = self.feature_extractor(x)
    a1 = torch.flatten(a1, 1)
    a2 = self.classifier(a1)
    return a2


def load_mnist_test_data(
    images_path: str, labels_path: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Loads, normalizes, and pads the MNIST 10k test dataset."""
  with open(images_path, "rb") as f:
    struct.unpack(">IIII", f.read(16))
    # Use .copy() to ensure the NumPy buffer is writable before converting to torch
    data = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(-1, 28, 28)

  # Normalize: (X / 255.0 - mean) / std
  norm = (data.astype(np.float32) / 255.0 - 0.1307) / 0.3081

  # Pad 2 border zeros on top, bottom, left, right to make 32x32 for LeNet
  padded = np.pad(
      norm, ((0, 0), (2, 2), (2, 2)), mode="constant", constant_values=0.0
  )
  images = padded[:, np.newaxis, :, :].copy()  # Shape: (N, 1, 32, 32)

  with open(labels_path, "rb") as f:
    struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

  return torch.from_numpy(images), torch.from_numpy(labels).long()


def evaluate_traced_model(images: torch.Tensor, labels: torch.Tensor) -> float:
  """Evaluates using traced_model.pt (no Python class definition needed)."""
  print(f"Loading traced model from {TRACED_MODEL_PATH}...")
  model = torch.jit.load(TRACED_MODEL_PATH, map_location="cpu")
  model.eval()

  with torch.no_grad():
    preds = model(images).argmax(dim=1)

  accuracy = (preds == labels).float().mean().item()
  return accuracy


def evaluate_state_dict_model(
    images: torch.Tensor, labels: torch.Tensor
) -> float:
  """Evaluates using model_state.pt (Option 2)."""
  print(f"Loading state dict from {MODEL_STATE_PATH}...")
  model = MultilayerPerceptron()
  state_dict = torch.load(MODEL_STATE_PATH, map_location="cpu")
  model.load_state_dict(state_dict)
  model.eval()

  with torch.no_grad():
    preds = model(images).argmax(dim=1)

  accuracy = (preds == labels).float().mean().item()
  return accuracy


if __name__ == "__main__":
  images_t, labels_t = load_mnist_test_data(IMAGES_PATH, LABELS_PATH)
  print(f"Loaded {len(labels_t)} test samples. Input shape: {images_t.shape}")

  # Evaluate using traced_model.pt
  if os.path.exists(TRACED_MODEL_PATH):
    traced_acc = evaluate_traced_model(images_t, labels_t)
    print(
        f"-> Traced Model Accuracy: {traced_acc:.4f} ({traced_acc * 100:.2f}%)"
    )

  # Evaluate using model_state.pt
  if os.path.exists(MODEL_STATE_PATH):
    state_dict_acc = evaluate_state_dict_model(images_t, labels_t)
    print(
        f"-> State Dict Model Accuracy: {state_dict_acc:.4f}"
        f" ({state_dict_acc * 100:.2f}%)"
    )
