from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class SoftmaxModel(nn.Module):

  def __init__(self):
    super(SoftmaxModel, self).__init__()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    return self.softmax(x)


model = SoftmaxModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
