from heir import compile
from heir.mlir import *

from absl.testing import absltest  # fmt: skip
class EndToEndTest(absltest.TestCase):

  def test_simple_arithmetic(self):
    @compile() # defaults to BGV and OpenFHE
    def foo(a : Secret[I16], b : Secret[I16]):
      return a * a - b * b

    # Test plaintext functionality
    self.assertEqual(-15, foo(7, 8))

    # Test FHE functionality
    foo.setup()
    enc_a = foo.encrypt_a(7)
    enc_b = foo.encrypt_b(8)
    result_enc = foo.foo(enc_a, enc_b)
    result = foo.decrypt_result(result_enc)
    self.assertEqual(-15, result)


if __name__ == "__main__":
  absltest.main()
