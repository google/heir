"""Common utilities for Torch CI scripts."""

import os
import torch_mlir
from torch_mlir.fx import OutputType


def export_and_save(model, sample_inputs, script_file):
  """Exports a torch model to MLIR and saves it to a file.

  Args:
      model: The torch.nn.Module to export.
      sample_inputs: A tuple of sample inputs for the model.
      script_file: The __file__ of the calling script, used to derive the output
        filename.
  """
  mlir = torch_mlir.fx.export_and_import(
      model, *sample_inputs, output_type=OutputType.LINALG_ON_TENSORS
  )

  script_dir = os.path.dirname(os.path.abspath(script_file))
  op_name = os.path.splitext(os.path.basename(script_file))[0]
  output_path = os.path.join(script_dir, f"{op_name}.mlir")

  with open(output_path, "w") as f:
    f.write(str(mlir))

  print(f"Exported MLIR to {output_path}")
