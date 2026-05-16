from scripts.torch_ci.common import export_and_save
import torch
import torch.nn as nn


class ChunkModel(nn.Module):

  def __init__(self):
    super(ChunkModel, self).__init__()

  def forward(self, x):
    return torch.chunk(x, chunks=2, dim=1)


model = ChunkModel()
model.eval()

# Example input
sample_input = torch.randn(1, 64)

# Export to MLIR
export_and_save(model, (sample_input,), __file__)
