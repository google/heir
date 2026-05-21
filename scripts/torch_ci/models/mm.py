from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class MMModel(nn.Module):

  def __init__(self):
    super(MMModel, self).__init__()

  def forward(self, x, y):
    return torch.mm(x, y)


model = MMModel()
model.eval()

# Example input
sample_input_x = torch.randn(64, 32)
sample_input_y = torch.randn(32, 10)

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
