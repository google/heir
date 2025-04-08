from heir import compile
from heir.mlir import I16, Secret, Tensor


@compile(
    debug=True
)  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(x: Secret[Tensor[1024, I16]], y: Secret[Tensor[1024, I16]]):
  result = x
  for i in range(10):
    result = result + y
  return result * result


x = [v for v in range(1024)]
y = [2 * v for v in range(1024)]
func(x, y)
