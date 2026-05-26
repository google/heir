import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class SqrtModel(nn.Module):

  def __init__(self):
    super(SqrtModel, self).__init__()

  def forward(self, x):
    return torch.sqrt(x)


model = SqrtModel()
model.eval()

# Example input (must be non-negative)
sample_input = torch.rand(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
