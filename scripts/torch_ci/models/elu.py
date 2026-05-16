from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class ELUModel(nn.Module):

  def __init__(self):
    super(ELUModel, self).__init__()
    self.elu = nn.ELU()

  def forward(self, x):
    return self.elu(x)


model = ELUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
