"""A demonstration of adding 1 to a number in FHE."""
from typing import Any

from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_bool import type_converters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import types


def setup_test_params() -> Any:
  """Set up the keys and encryptions for add_one."""
  boolean_params = bool_params.get_params_for_test()
  lwe_rng = bool_params.get_rng_for_test(1)
  rlwe_rng = bool_params.get_rng_for_test(1)
  cks = jaxite_bool.ClientKeySet(boolean_params, lwe_rng, rlwe_rng)
  sks = jaxite_bool.ServerKeySet(cks, boolean_params, lwe_rng, rlwe_rng)
  return (lwe_rng, boolean_params, cks, sks)


def encrypt_u8(
    x: int,
    cks: jaxite_bool.ClientKeySet,
    lwe_rng: random_source.RandomSource,
) -> list[types.LweCiphertext]:
  """Encrypt 8-bit integer."""
  cleartext_x = type_converters.u8_to_bit_slice(x)
  ciphertext_x = [jaxite_bool.encrypt(z, cks, lwe_rng) for z in cleartext_x]

  return ciphertext_x


def decrypt_u8(
    ciphertext: list[types.LweCiphertext], cks: jaxite_bool.ClientKeySet
) -> int:
  """Decrypt 8-bit integer."""
  return type_converters.bit_slice_to_u8(
      [jaxite_bool.decrypt(z, cks) for z in ciphertext]
  )
