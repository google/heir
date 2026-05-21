from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class LeakyReLUModel(nn.Module):

  def __init__(self):
    super(LeakyReLUModel, self).__init__()
    self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

  def forward(self, x):
    return self.leaky_relu(x)


model = LeakyReLUModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
