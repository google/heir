from heir import compile
from heir.backends.cleartext import CleartextBackend
from heir.mlir import F64, I1, I32, I64, Secret

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

  def test_comparison_float(self):
    # A comparison's result type is always bool (I1), which is narrower
    # than its operands by design. This used to be misclassified by the
    # same bitwidth-matching logic meant for arithmetic ops, and crashed
    # before ever reaching the (correct) comparison-handling code.

    @compile(scheme="ckks", backend=CleartextBackend())
    def cmp_float(x: Secret[F64], y: F64) -> I1:
      return x < y

    self.assertEqual(False, cmp_float.original(6.0, 2.0))
    self.assertEqual(True, cmp_float.original(2.0, 6.0))

  def test_comparison_int(self):

    @compile(scheme="ckks", backend=CleartextBackend())
    def cmp_int(x: Secret[I32], y: I32) -> I1:
      return x < y

    self.assertEqual(False, cmp_int.original(6, 2))
    self.assertEqual(True, cmp_int.original(2, 6))


if __name__ == "__main__":
  absltest.main()
