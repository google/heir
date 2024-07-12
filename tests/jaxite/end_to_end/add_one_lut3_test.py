"""Tests for add_one."""

from absl.testing import absltest
from heir.tests.jaxite.end_to_end import add_one_lut3_lib
from heir.tests.jaxite.end_to_end import test_utils


class AddOneLut3Test(absltest.TestCase):
  def test_add_one(self):
    x = 5
    lwe_rng, boolean_params, cks, sks = test_utils.setup_test_params()
    ciphertext_x = test_utils.encrypt_u8(x, cks, lwe_rng)

    result_ciphertext = add_one_lut3_lib.test_add_one_lut3(
        ciphertext_x,
        sks,
        boolean_params,
    )

    result = test_utils.decrypt_u8(result_ciphertext, cks)
    self.assertEqual(x + 1, result)


if __name__ == '__main__':
  absltest.main()
