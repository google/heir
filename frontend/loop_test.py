from heir import compile
from heir.mlir import Secret, I64

from absl.testing import absltest  # fmt: skip


class EndToEndTest(absltest.TestCase):

  def test_loop(self):
    @compile()
    def loopa(a: Secret[I64]):
      result = a
      lb = 1
      ub = 5
      for _ in range(lb, ub):
        result = result + result
      return result

    self.assertEqual(32, loopa(2))


if __name__ == "__main__":
  absltest.main()
