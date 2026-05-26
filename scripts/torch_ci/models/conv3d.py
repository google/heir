import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class Conv3dModel(nn.Module):

  def __init__(self):
    super(Conv3dModel, self).__init__()
    self.conv = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(x)


model = Conv3dModel()
model.eval()

# Example input: (batch_size, channels, depth, height, width)
sample_input = torch.randn(1, 3, 4, 4, 4)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
