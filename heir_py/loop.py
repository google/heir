"""Example of HEIR Python usage."""

from heir_py import pipeline


def foo(a):
  """An example function with a static loop."""
  result = 2
  for i in range(3):
    result = a + result
  return result


# to replace with decorator
_heir_foo = pipeline.run_compiler(foo).module

cc = _heir_foo.foo__generate_crypto_context()
kp = cc.KeyGen()
_heir_foo.foo__configure_crypto_context(cc, kp.secretKey)
arg0_enc = _heir_foo.foo__encrypt__arg0(cc, 2, kp.publicKey)
res_enc = _heir_foo.foo(cc, arg0_enc)
res = _heir_foo.foo__decrypt__result0(cc, res_enc, kp.secretKey)

print(res)  # should be 8
