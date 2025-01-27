from heir.mlir.mlir_decorator import mlir
from numba.core import sigutils
import numpy as np

## Example of an overload: matmul
@mlir
def matmul(typingctx, X, Y):
  # TODO (#1162): add a check if input types are valid!
  # FIXME: How to get the shape of the inputs via type inference?
  return sigutils._parse_signature_string("float32[:,:](float32[:,:],float32[:,:])"), np.matmul
