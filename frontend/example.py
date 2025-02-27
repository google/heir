"""Example of HEIR Python usage."""

from heir import compile
from heir.mlir import F32, I16, I64, Secret, Tensor

# TODO (#1162): Also add the tensorflow-to-tosa-to-HEIR example in example.py, even it doesn't use the main Python frontend?


### Simple Example
@compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
def func(x: Secret[I16], y: Secret[I16]):
  sum = x + y
  diff = x - y
  mul = x * y
  expression = sum * diff + mul
  deadcode = expression * mul
  return expression


print(
    f"Expected result for `func`: {func.original(7,8)}, FHE result: {func(7,8)}"
)


# ### Manual setup/enc/dec example
# @compile()
# def foo(x: Secret[I16], y: Secret[I16]):
#   return (x + y) * (x - y) + (x * y)


# foo.setup()  # runs keygen/etc
# enc_x = foo.encrypt_x(7)
# enc_y = foo.encrypt_y(8)
# result_enc = foo.eval(enc_x, enc_y)
# result = foo.decrypt_result(result_enc)
# print(
#     f"Expected result for `foo`: {foo.original(7,8)}, "
#     f"decrypted FHE result: {result}"
# )


# ### Loop Example
# @compile()
# def loop_test(a: Secret[I64]):
#   """An example function with a static loop."""
#   result = 2
#   for i in range(3):
#     result = a + result
#   return result


# print(
#     f"Expected result for `loop_test`: {loop_test(2)}, "
#     f"FHE result: {loop_test(2)}"
# )


# ### CKKS Example
# @compile(scheme="ckks")
# def bar(x: Secret[F32], y: Secret[F32]):
#   return (x + y) * (x - y) + (x * y)


# print(f"Expected result for `bar`: {bar.original(7,8)}, FHE result: {bar(7,8)}")


# ### Ciphertext-Plaintext Example
# @compile(debug=True)
# def baz(x: Secret[I16], y: Secret[I16], z: I16):
#   ptxt_mul = x * z
#   ctxt_mul = x * x
#   ctxt_mul2 = y * y
#   add = ctxt_mul + ctxt_mul2
#   return ptxt_mul + add


# print(
#     f"Expected result for `baz`: {baz.original(7,8,9)}, "
#     f"FHE result: {baz(7,8,9)}"
# )


# ### Custom Pipeline Example
# @compile(
#     heir_opt_options=["--mlir-to-secret-arithmetic", "--canonicalize", "--cse"],
#     backend=None,  # defaults to CleartextBackend
#     debug=True,
# )
# def custom(x: Secret[I16], y: Secret[I16]):
#   return (x + y) * (x - y) + (x * y)


# print(
#     f"CleartextBackend simply runs the original python function: {custom(7,8)}"
# )
