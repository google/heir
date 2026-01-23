from heir import compile
from heir.mlir import I16, I32, Secret

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

  def test_literal_extension(self):

    @compile()
    def foo(x: Secret[I32]):
      a = 45  # 45 fits inside int8
      b = 10 * a  # 10 fits inside int8
      # but 450 does not, so we must upcast
      return x + b

    self.assertEqual(455, foo(5))

  def test_intentional_overflow(self):

    @compile()
    def foo(x: Secret[I16]):
      a = 32762  # 32762 fits inside int16
      b = a + 5  # so does 32767 (barely)
      return x + b  # but this will overflow for non-zero x

    self.assertEqual(-(2**15), foo(1))

  def test_literal_variable_extension(self):

    @compile()
    def foo(x: Secret[I32]):
      a = x * 10  # i32 * i8 -> i32
      return a + 5

    self.assertEqual(455, foo(45))


if __name__ == "__main__":
  absltest.main()
