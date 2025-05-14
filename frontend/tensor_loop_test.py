from heir import compile
from heir.mlir import I16, Secret, Tensor
import numpy as np
from numpy.testing import assert_array_equal


from absl.testing import absltest  # fmt: skip
class TensorLoopTest(absltest.TestCase):

  def test_tensor_loop(self):

    @compile()  # defaults to scheme="bgv", OpenFHE backend, and debug=False
    def func(x: Secret[Tensor[[1024], I16]], y: Secret[Tensor[[1024], I16]]):
      result = x
      for i in range(10):
        result = result + y
      return result * result

    a = np.array([7] * 1024, dtype=np.int16)
    b = np.array([8] * 1024, dtype=np.int16)
    expected = (a + 10 * b) ** 2

    # Test cleartext functionality
    assert_array_equal(expected, func.original(a, b))

    # Test FHE functionality
    assert_array_equal(expected, func(a, b))


if __name__ == "__main__":
  absltest.main()
