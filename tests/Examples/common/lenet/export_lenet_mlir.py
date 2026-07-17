"""Export a trained LeNet model to Linalg MLIR using torch_mlir.

python3 export_lenet_mlir.py
--checkpoint_dir=third_party/heir/tests/Examples/common/lenet
--output_mlir=third_party/heir/tests/Examples/common/lenet/lenet.mlir
"""

import argparse
import os
import sys
import torch
from torch import nn
from torch.nn import functional as F
import torch_mlir
from torch_mlir.fx import OutputType


class MultilayerPerceptron(nn.Module):
  """Self-contained LeNet-5 architecture matching model.py."""

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


def main():
  parser = argparse.ArgumentParser(
      description="Export trained LeNet (MultilayerPerceptron) to Linalg MLIR."
  )
  parser.add_argument(
      "--checkpoint_dir",
      type=str,
      required=True,
      help="Directory containing trained model_state.pt",
  )
  parser.add_argument(
      "--output_mlir",
      type=str,
      default=None,
      help="Output path for lenet.mlir (defaults to checkpoint_dir/lenet.mlir)",
  )
  args = parser.parse_args()

  checkpoint_dir = args.checkpoint_dir
  output_mlir = args.output_mlir or os.path.join(checkpoint_dir, "lenet.mlir")

  state_path = os.path.join(checkpoint_dir, "model_state.pt")

  mlp = MultilayerPerceptron()
  if os.path.exists(state_path):
    print(f"Loading trained weights from state dict: {state_path}")
    state_dict = torch.load(state_path, map_location="cpu")
    mlp.load_state_dict(state_dict)
  else:
    print(
        f"Error: {state_path} does not exist in {checkpoint_dir}.",
        file=sys.stderr,
    )
    sys.exit(1)

  mlp.eval()

  # LeNet expects 1x1x32x32 sample input
  sample_input = torch.zeros(1, 1, 32, 32, dtype=torch.float32)

  print("Exporting MultilayerPerceptron to Linalg on Tensors (lenet.mlir)...")
  mlir_module = torch_mlir.fx.export_and_import(
      mlp, sample_input, output_type=OutputType.LINALG_ON_TENSORS
  )

  os.makedirs(os.path.dirname(os.path.abspath(output_mlir)), exist_ok=True)
  with open(output_mlir, "w") as f:
    f.write(str(mlir_module))

  print(f"Successfully exported {output_mlir}")


if __name__ == "__main__":
  main()
