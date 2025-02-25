"""Tests for fully_connected."""

from absl.testing import absltest
from tests.Examples.jaxite import fully_connected_lib
from tests.Examples.jaxite import test_utils


class FullyConnectedTest(absltest.TestCase):

  def test_add_one(self):
    x = 25
    lwe_rng, boolean_params, cks, sks = test_utils.setup_test_params()
    ciphertext_x = test_utils.encrypt_u8(x, cks, lwe_rng)

    result_ciphertext = fully_connected_lib.main(
        ciphertext_x,
        sks,
        boolean_params,
    )

    result = test_utils.decrypt_int(result_ciphertext, cks, num_bits=32)
    # The result should be x + 1 + 128 (input_zp = -128)
    self.assertEqual(x + 1 + 128, result)


if __name__ == "__main__":
  absltest.main()
