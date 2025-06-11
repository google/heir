from heir import compile
from heir.mlir import I8, Secret
from heir.backends.cleartext import CleartextBackend


from absl.testing import absltest  # fmt: skip
class EndToEndTest(absltest.TestCase):

  def test_simple_cggi_arithmetic(self):

    @compile(
        scheme="cggi",
        backend=CleartextBackend(),
        debug="True",
    )
    def foo(a: Secret[I8], b: Secret[I8]):
      return a * a - b * b

    # Test cleartext functionality
    self.assertEqual(-15, foo.original(7, 8))

    # Test FHE functionality
    foo.setup()
    enc_a = foo.encrypt_a(7)
    enc_b = foo.encrypt_b(8)
    result_enc = foo.eval(enc_a, enc_b)
    result = foo.decrypt_result(result_enc)
    self.assertEqual(-15, result)


if __name__ == "__main__":
  absltest.main()
