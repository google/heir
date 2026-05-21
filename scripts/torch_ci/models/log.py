from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class LogModel(nn.Module):

  def __init__(self):
    super(LogModel, self).__init__()

  def forward(self, x):
    return torch.log(x)


model = LogModel()
model.eval()

# Example input (must be positive for log)
sample_input = torch.rand(1, 64) + 0.1

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
