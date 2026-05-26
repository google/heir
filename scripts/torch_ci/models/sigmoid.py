import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class SigmoidModel(nn.Module):

  def __init__(self):
    super(SigmoidModel, self).__init__()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    return self.sigmoid(x)


model = SigmoidModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
