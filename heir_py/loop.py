"""Example of HEIR Python usage."""

from heir_py import pipeline


def foo(a):
  """An example function with a static loop."""
  result = 0
  for i in range(3):
    result = a * result
  return result

'''
func.func @foo(%arg0: i64) -> i64 {
  %c0_i16 = arith.constant 0 : i64
  %0 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %c0_i16) -> (i64) {
    %5 = arith.muli %arg3, %arg0 : i64
    affine.yield %5 : i64
  }
  return %0 : i64
}
'''

# to replace with decorator
_heir_foo = pipeline.run_compiler(foo)

cc = _heir_foo.foo__generate_crypto_context()
kp = cc.KeyGen()
_heir_foo.foo__configure_crypto_context(cc, kp.secretKey)
arg0_enc = _heir_foo.foo__encrypt__arg0(cc, 2, kp.publicKey)
res_enc = _heir_foo.foo(cc, arg0_enc, arg1_enc)
res = _heir_foo.foo__decrypt__result0(cc, res_enc, kp.secretKey)

print(res)  # should be 8
