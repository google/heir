import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class LinearModel(nn.Module):

  def __init__(self):
    super(LinearModel, self).__init__()
    self.linear = nn.Linear(64, 10)

  def forward(self, x):
    return self.linear(x)


model = LinearModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
