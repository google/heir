from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class DivModel(nn.Module):

  def __init__(self):
    super(DivModel, self).__init__()

  def forward(self, x, y):
    return x / y


model = DivModel()
model.eval()

# Example input (y must not be zero)
sample_input_x = torch.randn(1, 64)
sample_input_y = torch.randn(1, 64) + 1.0  # Avoid zero

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
