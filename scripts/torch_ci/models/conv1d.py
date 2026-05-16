from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class Conv1dModel(nn.Module):

  def __init__(self):
    super(Conv1dModel, self).__init__()
    self.conv = nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(x)


model = Conv1dModel()
model.eval()

# Example input: (batch_size, channels, length)
sample_input = torch.randn(1, 3, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
