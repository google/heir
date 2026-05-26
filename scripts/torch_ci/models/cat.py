import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class CatModel(nn.Module):

  def __init__(self):
    super(CatModel, self).__init__()

  def forward(self, x, y):
    return torch.cat((x, y), dim=1)


model = CatModel()
model.eval()

# Example input
sample_input_x = torch.randn(1, 32, 16, 16)
sample_input_y = torch.randn(1, 32, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
