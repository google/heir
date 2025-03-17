from heir import compile
from heir.mlir import I1, I64, Secret

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_cond(self):

    @compile(debug=True)
    def cond(a: Secret[I64], b: Secret[I1]):
      result = 0
      if b:
        result = a
      else:
        result = 0
      result = result + 1
      return result

    self.assertEqual(2, cond(2))


if __name__ == "__main__":
  absltest.main()
