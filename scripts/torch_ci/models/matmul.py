import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class MatmulModel(nn.Module):

  def __init__(self):
    super(MatmulModel, self).__init__()

  def forward(self, x, y):
    return torch.matmul(x, y)


model = MatmulModel()
model.eval()

# Example input
sample_input_x = torch.randn(1, 64)
sample_input_y = torch.randn(64, 10)

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
