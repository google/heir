import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class MeanModel(nn.Module):

  def __init__(self):
    super(MeanModel, self).__init__()

  def forward(self, x):
    return torch.mean(x, dim=1)


model = MeanModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
