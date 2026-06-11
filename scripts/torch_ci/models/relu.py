import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class ReLUModel(nn.Module):

  def __init__(self):
    super(ReLUModel, self).__init__()
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.relu(x)


model = ReLUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
