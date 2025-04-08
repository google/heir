from heir import compile
from heir.mlir import I16, Secret, Tensor


from absl.testing import absltest  # fmt: skip
class TensorTest(absltest.TestCase):

  def test_basic_tensors(self):

    @compile()  # defaults to BGV and OpenFHE
    def foo(a: Secret[Tensor[1024, I16]], b: Secret[Tensor[1024, I16]]):
      return a * a - b * b

    # Test cleartext functionality
    self.assertEqual(-15, foo.original(7, 8))

    # Test FHE functionality
    self.assertEqual([-15] * 1024, foo([7] * 1024, [8] * 1024))


if __name__ == "__main__":
  absltest.main()
