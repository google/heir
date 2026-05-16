from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class Conv2dModel(nn.Module):

  def __init__(self):
    super(Conv2dModel, self).__init__()
    self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(x)


model = Conv2dModel()
model.eval()

# Example input: (batch_size, channels, height, width)
sample_input = torch.randn(1, 3, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
