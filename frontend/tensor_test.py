from heir import compile
from heir.mlir import I16, Secret, Tensor
import numpy as np
from numpy.testing import assert_array_equal


from absl.testing import absltest  # fmt: skip
class TensorTest(absltest.TestCase):

  def test_basic_tensors(self):

    @compile()  # defaults to BGV and OpenFHE
    def foo(a: Secret[Tensor[[1024], I16]], b: Secret[Tensor[[1024], I16]]):
      return a * a - b * b

    a = np.array([7] * 1024, dtype=np.int16)
    b = np.array([8] * 1024, dtype=np.int16)
    expected = a * a - b * b

    # Test cleartext functionality
    assert_array_equal(expected, foo.original(a, b))

    # Test FHE functionality
    assert_array_equal(expected, foo(a, b))


if __name__ == "__main__":
  absltest.main()
