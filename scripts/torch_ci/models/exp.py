import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class ExpModel(nn.Module):

  def __init__(self):
    super(ExpModel, self).__init__()

  def forward(self, x):
    return torch.exp(x)


model = ExpModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
