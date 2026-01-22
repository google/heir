from heir import compile
from heir.mlir import I32, Secret

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_mixed_bitwidth(self):

    @compile()
    def foo(a: Secret[I32]):
      b = 5
      return a + b

    self.assertEqual(10, foo(5))

  def test_signed_cast(self):

    @compile()
    def foo(x: Secret[I32]):
      return -1 * x

    self.assertEqual(-5, foo(5))


if __name__ == "__main__":
  absltest.main()
