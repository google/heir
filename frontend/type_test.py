from heir import compile
from heir.mlir import I64, Secret

from absl.testing import absltest  # fmt: skip


class TypeTest(absltest.TestCase):

  def test_type(self):

    @compile()
    def type_two(x: Secret[I64], y: Secret[I64]):
      xsquare = x * x
      # The constant 2 is i32 typed, but needs to be sign extended to an i64
      twox = 2 * x
      first = xsquare - twox
      result = first + y
      return result

    # 2*2 - 2*2 + 3 = 3
    self.assertEqual(3, type_two(2, 3))


if __name__ == "__main__":
  absltest.main()
