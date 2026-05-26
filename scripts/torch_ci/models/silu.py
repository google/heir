import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class SiLUModel(nn.Module):

  def __init__(self):
    super(SiLUModel, self).__init__()
    self.silu = nn.SiLU()

  def forward(self, x):
    return self.silu(x)


model = SiLUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
