import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class AvgPool1dModel(nn.Module):

  def __init__(self):
    super(AvgPool1dModel, self).__init__()
    self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.avgpool(x)


model = AvgPool1dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 4, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
