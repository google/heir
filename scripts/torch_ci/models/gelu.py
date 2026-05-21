from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class GELUModel(nn.Module):

  def __init__(self):
    super(GELUModel, self).__init__()
    self.gelu = nn.GELU()

  def forward(self, x):
    return self.gelu(x)


model = GELUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
