from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class MaxPool1dModel(nn.Module):

  def __init__(self):
    super(MaxPool1dModel, self).__init__()
    self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.maxpool(x)


model = MaxPool1dModel()
model.eval()

# Example input
sample_input = torch.randn(1, 3, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
