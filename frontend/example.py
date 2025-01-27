"""Example of HEIR Python usage."""
# This file can be run from the main repo folder as `python -m heir_py.example`

from heir import compile
from heir.mlir import *

# FIXME: Also add the tensorflow-to-tosa-to-HEIR example in here, even it doesn't use Numba
# TODO (#1162): Allow manually specifying precision/scale and warn/error if not possible!
# TODO (#????): Add precision computation/check to Mgmt dialect/infrastructure
# TODO (#1119): expose ctxt serialization in python
# TODO (#1162) : Fix "ImportError: generic_type: type "PublicKey" is already registered!" when doing setup twice. (Required to allow multiple compilations in same python file)
# TODO (#????): In OpenFHE is it more efficient to do mul(...) WITH relin than to do mul_no_relin and then relin? If yes, write a simple peephhole opt to rewrite this
# TODO (HECO OPT): Do not touch loops that are already operating on tensors/SIMD values
# TODO (#1162): Check if we prevent people from doing a = enc_x, b = enc_y for foo(x,y) but then calling foo(b,a)!
# TODO (#1162): Allow for multiple functions in the same compilation. This requires switching from a decorator to a context thingy (`with heir.compile(...):`)
# TODO (#1162): Consider hardening the eval function against handcrafted ciphertexts with wrong metadata being passed in?

### Simple Example
@compile(debug=True) # defaults to scheme="bgv", backend="openfhe", debug=False
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


# ### CKKS Example
# @compile(scheme="ckks") # other options default to backend="openfhe", debug=False
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
# @compile(debug=True)
# def baz2(x: Secret[I16], y : Secret[I16], z : I16):
#   ptxt_mul = x * z
#   ctxt_mul = x * x
#   ctxt_mul2 = y * y
#   add = ctxt_mul + ctxt_mul2
#   return  ptxt_mul + add
#   # TODO (#1284): Relin Opt works if ptxt_mul is RHS, but not if ptxt_mul is LHS?

# baz2.setup() # runs keygen/etc
# enc_x = baz2.encrypt_x(7)
# enc_y = baz2.encrypt_y(8)
# result_enc = baz2.eval(enc_x, enc_y, 9)
# result = baz2.decrypt_result(result_enc)
# print(f"Expected result for `baz2`: {baz2(7,8,9)}, decrypted result: {result}")


# ### Custom Pipeline Example
# @compile(heir_opt_options=["--mlir-to-openfhe-bgv", "--canonicalize", "--cse"], backend=None, debug=True)
# def foo(x : Secret[I16], y : Secret[I16]):
#   sum = x + y
#   diff = x - y
#   mul = x * y
#   expression = sum * diff + mul
#   deadcode = expression * mul
#   return expression


# ### Heracles SDK Backend Example
# @compile(backend="heracles", debug=True)
# def foo(x : Secret[I16], y : Secret[I16], z: I16):
#   sum = x + z
#   diff = x - y
#   mul = x * y
#   expression = sum * diff + mul
#   return expression

# ### Matmul Example
# # secret.secret<tensor<4x4xf32>>
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
