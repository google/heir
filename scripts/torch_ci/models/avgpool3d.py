from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class AvgPool3dModel(nn.Module):

  def __init__(self):
    super(AvgPool3dModel, self).__init__()
    self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.avgpool(x)


model = AvgPool3dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 4, 4, 4, 4)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
