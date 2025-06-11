from heir import compile
from heir.mlir import I1, I8, Secret
from heir.backends.cleartext import CleartextBackend


from absl.testing import absltest  # fmt: skip
class EndToEndTest(absltest.TestCase):

  def test_cggi_cast(self):

    @compile(
        scheme="cggi",
        backend=CleartextBackend(),
        debug=True,
    )
    def foo(x: Secret[I8]):
      x0 = I1((x >> 7) & 1)
      return x0

    # Test cleartext functionality
    self.assertEqual(1, foo.original(255))
    self.assertEqual(0, foo.original(16))

    # Test FHE functionality
    self.assertEqual(1, foo(255))
    self.assertEqual(0, foo(16))


if __name__ == "__main__":
  absltest.main()
