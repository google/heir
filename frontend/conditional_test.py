from heir import compile
from heir.mlir import I64, Secret

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_cond(self):

    @compile(debug=True)
    def cond(a: Secret[I64]):
      result = 0
      if a == 0:
        result = result + 1
        # return result
      else:
        result = 2
      result = result + 1
      return result

    self.assertEqual(2, cond(2))


if __name__ == "__main__":
  absltest.main()
