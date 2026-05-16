from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class AvgPool2dModel(nn.Module):

  def __init__(self):
    super(AvgPool2dModel, self).__init__()
    self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.avgpool(x)


model = AvgPool2dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 4, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
