"""Example of HEIR Python usage."""

from heir import compile, compile_mlir
from heir.mlir import F32, I16, I64, Secret
from heir.backends.cleartext import CleartextBackend

# TODO (#1162): Also add the tensorflow-to-tosa-to-HEIR example in example.py, even it doesn't use the main Python frontend?

# TODO (#1162): Remove the need for wrapper functions around each `@compile`-d function to isolate backend pybindings


### Simple Example
def simple_example():
  print("Running simple example")

  @compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
  def func(x: Secret[I16], y: Secret[I16]):
    sum = x + y
    diff = x - y
    mul = x * y
    expression = sum * diff + mul
    deadcode = expression * mul
    return expression

  print(
      f"Expected result for `func`: {func.original(7,8)}, FHE result:"
      f" {func(7,8)}"
  )


### Manual setup/enc/dec example
def manual_example():
  print("Running manual example")

  @compile()
  def foo(x: Secret[I16], y: Secret[I16]):
    return (x + y) * (x - y) + (x * y)

  foo.setup()  # runs keygen/etc
  enc_x = foo.encrypt_x(7)
  enc_y = foo.encrypt_y(8)
  result_enc = foo.eval(enc_x, enc_y)
  result = foo.decrypt_result(result_enc)
  print(
      f"Expected result for `foo`: {foo.original(7,8)}, "
      f"decrypted FHE result: {result}"
  )


### Loop Example
def loop_example():
  print("Running loop example")

  @compile()
  def loop_test(a: Secret[I64]):
    """An example function with a static loop."""
    result = 2
    for i in range(3):
      result = a + result
    return result

  print(
      f"Expected result for `loop_test`: {loop_test(2)}, "
      f"FHE result: {loop_test(2)}"
  )


### CKKS Example
def ckks_example():
  print("Running CKKS example")

  @compile(scheme="ckks")
  def bar(x: Secret[F32], y: Secret[F32]):
    return (x + y) * (x - y) + (x * y)

  print(
      f"Expected result for `bar`: {bar.original(0.7,0.8)}, FHE result:"
      f" {bar(0.7,0.8)}"
  )


### Ciphertext-Plaintext Example
def ctxt_ptxt_example():
  print("Running ciphertext-plaintext example")

  @compile()
  def baz(x: Secret[I16], y: Secret[I16], z: I16):
    ptxt_mul = x * z
    ctxt_mul = x * x
    ctxt_mul2 = y * y
    add = ctxt_mul + ctxt_mul2
    return ptxt_mul + add

  print(
      f"Expected result for `baz`: {baz.original(7,8,9)}, "
      f"FHE result: {baz(7,8,9)}"
  )


### Custom Pipeline Example
def custom_example():
  print("Running custom pipeline example")

  @compile(
      heir_opt_options=[
          "--mlir-to-secret-arithmetic",
          "--canonicalize",
          "--cse",
      ],
      backend=CleartextBackend(),  # just runs the python function when `custom(...)` is called
      debug=True,  # so that we can see the file that contains the output of the pipeline
  )
  def custom(x: Secret[I16], y: Secret[I16]):
    return (x + y) * (x - y) + (x * y)

  print(
      "CleartextBackend simply runs the original python function:"
      f" {custom(7,8)}"
  )


def main():
  simple_example()
  manual_example()
  loop_example()
  ckks_example()
  ctxt_ptxt_example()
  custom_example()

  # MLIR-as-string example
  def mlir_example():
    print("Running MLIR-as-string example")

    mlir_src = """
func.func @myfunc() -> i32 {
  %c = arith.constant 42 : i32
  return %c : i32
}
"""

    client = compile_mlir(
        mlir_src,
        func_name="myfunc",
        arg_names=[],
        secret_args=[],
    )
    client.setup()
    print(f"Result: {client()}")

  mlir_example()


if __name__ == "__main__":
  main()
