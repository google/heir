import torch
import torch.nn as nn
from scripts.torch_ci.common import export_and_save


class NegModel(nn.Module):

  def __init__(self):
    super(NegModel, self).__init__()

  def forward(self, x):
    return -x


model = NegModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
