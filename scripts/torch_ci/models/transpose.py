from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class TransposeModel(nn.Module):

  def __init__(self):
    super(TransposeModel, self).__init__()

  def forward(self, x):
    return torch.transpose(x, 1, 2)


model = TransposeModel()
model.eval()

# Example input
sample_input = torch.randn(1, 3, 16, 16)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
