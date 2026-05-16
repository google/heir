from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class TanhModel(nn.Module):

  def __init__(self):
    super(TanhModel, self).__init__()
    self.tanh = nn.Tanh()

  def forward(self, x):
    return self.tanh(x)


model = TanhModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
