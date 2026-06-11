import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class BatchNorm2dModel(nn.Module):

  def __init__(self):
    super(BatchNorm2dModel, self).__init__()
    self.bn = nn.BatchNorm2d(3)

  def forward(self, x):
    return self.bn(x)


model = BatchNorm2dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 3, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
