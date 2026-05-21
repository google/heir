from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class SubModel(nn.Module):

  def __init__(self):
    super(SubModel, self).__init__()

  def forward(self, x, y):
    return x - y


model = SubModel()
model.eval()

# Example input
sample_input_x = torch.randn(1, 64)
sample_input_y = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
