import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class BMMModel(nn.Module):

  def __init__(self):
    super(BMMModel, self).__init__()

  def forward(self, x, y):
    return torch.bmm(x, y)


model = BMMModel()
model.eval()

# Example input: (batch_size, n, m), (batch_size, m, p)
sample_input_x = torch.randn(2, 64, 32)
sample_input_y = torch.randn(2, 32, 10)

# Export to MLIR
export_and_save(model, (sample_input_x, sample_input_y), __file__)
