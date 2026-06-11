import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class SqueezeModel(nn.Module):

  def __init__(self):
    super(SqueezeModel, self).__init__()

  def forward(self, x):
    return torch.squeeze(x)


model = SqueezeModel()
model.eval()

# Example input with a dimension of size 1
sample_input = torch.randn(1, 3, 1, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
