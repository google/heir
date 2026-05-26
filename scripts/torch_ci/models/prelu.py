import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class PReLUModel(nn.Module):

  def __init__(self):
    super(PReLUModel, self).__init__()
    self.prelu = nn.PReLU()

  def forward(self, x):
    return self.prelu(x)


model = PReLUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
