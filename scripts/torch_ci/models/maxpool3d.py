from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class MaxPool3dModel(nn.Module):

  def __init__(self):
    super(MaxPool3dModel, self).__init__()
    self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.maxpool(x)


model = MaxPool3dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 3, 4, 4, 4)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
