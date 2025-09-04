from absl.testing import absltest
from heir import compile
from heir.mlir import I32, Secret


class UniqueNameTest(absltest.TestCase):
  """This test ensures there are no name conflicts across lexical scopes."""

  def test_one(self):
    @compile()
    def add(x: Secret[I32], y: Secret[I32]):
      return x + y

    self.assertEqual(add(1, 2), 3)

  def test_two(self):
    @compile()
    def add(x: Secret[I32], y: Secret[I32]):
      return x * y

    self.assertEqual(add(2, 3), 6)


if __name__ == "__main__":
  absltest.main()
