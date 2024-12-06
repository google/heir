from heir_py import compile

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_simple_arithmetic(self):
    @compile(backend="openfhe")
    def foo(a, b):
      return a * a - b * b

    # Test the e2e path
    self.assertEqual(-15, foo(7, 8))

    # Also test the manual way with crypto_context and keys threaded in
    # automatically
    foo.setup()
    enc_a = foo.encrypt_a(7)
    enc_b = foo.encrypt_b(8)
    result_enc = foo.foo(enc_a, enc_b)
    result = foo.decrypt_result(result_enc)
    self.assertEqual(-15, result)


if __name__ == "__main__":
  absltest.main()
