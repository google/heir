import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class PermuteModel(nn.Module):

  def __init__(self):
    super(PermuteModel, self).__init__()

  def forward(self, x):
    return x.permute(0, 2, 3, 1)  # NCHW to NHWC


model = PermuteModel()
model.eval()

# Example input
sample_input = torch.randn(1, 3, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
