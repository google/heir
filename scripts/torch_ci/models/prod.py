from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class ProdModel(nn.Module):

  def __init__(self):
    super(ProdModel, self).__init__()

  def forward(self, x):
    return torch.prod(x, dim=1)


model = ProdModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
