"""Minimal (dependency-free) test of PyPI for cibuildwheel."""

from heir import compile
from heir.backends.cleartext import CleartextBackend
from heir.mlir import I16, Secret


# Custom Pipeline Example
def custom_example():
  print("Running custom pipeline example")

  @compile(
      heir_opt_options=[
          "--mlir-to-secret-arithmetic",
          "--canonicalize",
          "--cse",
      ],
      backend=CleartextBackend(),
      debug=True,
  )
  def custom(x: Secret[I16], y: Secret[I16]):
    return (x + y) * (x - y) + (x * y)


# MLIR-input example
def mlir_example():
  print("Running MLIR-input example")
  # Input should be a single function, just like normal MLIR input to heir-opt
  src = """
    func.func @myfunc(%a : i32 {secret.secret}, %b : i32) -> i32 {
      %sum = arith.addi %a, %b : i32
      return %sum : i32
    }
  """
  # By passing mlir_str instead of using it to decorate a function,
  # we can skip the python parsing/type inference/etc stages.
  compile(
      mlir_str=src,
      scheme="bgv",
      backend=CleartextBackend(),
      debug=True,
  )


def main():
  custom_example()
  mlir_example()


if __name__ == "__main__":
  main()
