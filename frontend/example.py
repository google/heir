"""Example of HEIR Python usage."""

from heir import compile
from heir.mlir import *

# TODO (#1162): Also add the tensorflow-to-tosa-to-HEIR example in example.py, even it doesn't use the main Python frontend?

### Simple Example
@compile() # defaults to scheme="bgv", OpenFHE backend, and debug=False
def foo(x : Secret[I16], y : Secret[I16]):
  sum = x + y
  diff = x - y
  mul = x * y
  expression = sum * diff + mul
  deadcode = expression * mul
  return expression

foo.setup() # runs keygen/etc
enc_x = foo.encrypt_x(7)
enc_y = foo.encrypt_y(8)
result_enc = foo.eval(enc_x, enc_y)
result = foo.decrypt_result(result_enc)
print(f"Expected result for `foo`: {foo(7,8)}, decrypted result: {result}")


# TODO (#1162) : Fix "ImportError: generic_type: type "PublicKey" is already registered!" when doing setup twice. (Required to allow multiple compilations in same python file)
### CKKS Example
# @compile(scheme="ckks")
# def bar(x : Secret[F32], y : Secret[F32]):
#   sum = x + y
#   diff = x - y
#   mul = x * y
#   expression = sum * diff + mul
#   deadcode = expression * mul
#   return expression

# bar.setup() # runs keygen/etc
# enc_x = bar.encrypt_x(7)
# enc_y = bar.encrypt_y(8)
# result_enc = bar.eval(enc_x, enc_y)
# result = bar.decrypt_result(result_enc)
# print(f"Expected result for `bar`: {bar(7,8)}, decrypted result: {result}")


# ### Ciphertext-Plaintext Example
# @compile()
# def baz2(x: Secret[I16], y : Secret[I16], z : I16):
#   ptxt_mul = x * z
#   ctxt_mul = x * x
#   ctxt_mul2 = y * y
#   add = ctxt_mul + ctxt_mul2
#   return  ptxt_mul + add

# baz2.setup() # runs keygen/etc
# enc_x = baz2.encrypt_x(7)
# enc_y = baz2.encrypt_y(8)
# result_enc = baz2.eval(enc_x, enc_y, 9)
# result = baz2.decrypt_result(result_enc)
# print(f"Expected result for `baz2`: {baz2(7,8,9)}, decrypted result: {result}")



# ### Custom Pipeline Example
# @compile(heir_opt_options=["--mlir-to-secret-arithmetic", "--canonicalize", "--cse"], backend=None, debug=True)
# def foo(x : Secret[I16], y : Secret[I16]):
#   sum = x + y
#   diff = x - y
#   mul = x * y
#   expression = sum * diff + mul
#   deadcode = expression * mul
#   return expression

# # The below are basically no-ops/plain python with the Dummy Backend
# foo.setup()
# enc_x = foo.encrypt_x(7)
# enc_y = foo.encrypt_y(8)
# result_enc = foo.eval(enc_x, enc_y)
# result = foo.decrypt_result(result_enc)
# print(f"Expected result for `foo`: {foo(7,8)}, decrypted result: {result}")


# ### Matmul Example (WIP)
# # #TODO (#1330): Implement Shape Inference and support, e.g., matmul in Frontend)
# from heir.mlir.linalg import matmul
# import numpy as np
# @compile(scheme='ckks', debug=True)
# def qux(a : Secret[Tensor[4,4,F32]], b : Secret[Tensor[4,4,F32]]):
#  AB = matmul(a,b)
#  AABB = matmul(a+a, b+b)
#  return AB + AABB

# a = np.array([[1,2],[3,4]])
# b = np.array([[5,6],[7,8]])
# print(qux(a,b))

# qux.setup()
# enc_a = qux.encrypt_a(a)
# enc_b = qux.encrypt_b(b)
# result_enc = qux.eval(enc_a, enc_b)
# result = qux.decrypt_result(result_enc)
# print(f"Expected result: {np.matmul(a,b)}, decrypted result: {result}")
