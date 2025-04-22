from heir import compile
from heir.mlir import I1, I64, Secret

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_cond(self):

    @compile(debug=True)
    def cond(cond1: Secret[I1], a: I64, b: I64):
      result = 0
      if cond1:
        result = a
      else:
        result = b
      return result

    self.assertEqual(2, cond(1, 2, 4))
    self.assertEqual(4, cond(0, 2, 4))


if __name__ == "__main__":
  absltest.main()
