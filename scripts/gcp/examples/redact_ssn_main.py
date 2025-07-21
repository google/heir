r"""A string redaction example to redact SSN from a string."""

from collections.abc import Sequence

from absl import app
import timeit

from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_bool import type_converters
from jaxite.jaxite_lib import random_source
from jaxite.jaxite_lib import test_utils
from jaxite.jaxite_lib import types

import redact_ssn_fhe_py_lib_p1

INPUT_STRING = 'dont log my ssn 123-45-6789'
# Note: This length should be same as the size used to generate the circuit in
# string_cap_fhe_lib.
STRING_SIZE = 32


def redact_ssn(
    my_string: list[types.LweCiphertext],
    sks: jaxite_bool.ServerKeySet,
    params: jaxite_bool.Parameters,
) -> list[types.LweCiphertext]:
  """Capitalizer first letter of every word in a string."""
  return redact_ssn_fhe_py_lib_p1.redact_ssn(my_string, sks, params)


def setup_test_params_keys() -> tuple[
    jaxite_bool.Parameters,
    random_source.PseudorandomSource,
    jaxite_bool.ClientKeySet,
    jaxite_bool.ServerKeySet,
]:
  """Perform necessary setup to generate parameters and keys for unit tests."""
  # Setup boolean scheme parameters
  boolean_params = bool_params.get_params_for_test()
  lwe_rng = bool_params.get_rng_for_test(1)
  rlwe_rng = bool_params.get_rng_for_test(1)
  cks = jaxite_bool.ClientKeySet(boolean_params, lwe_rng, rlwe_rng)
  sks = jaxite_bool.ServerKeySet(cks, boolean_params, lwe_rng, rlwe_rng)
  return (boolean_params, lwe_rng, cks, sks)


def setup_params_keys(
    decrypt_mid_bootstrap: bool = False,
) -> tuple[
    jaxite_bool.Parameters,
    random_source.PseudorandomSource,
    jaxite_bool.ClientKeySet,
    jaxite_bool.ServerKeySet,
]:
  """Perform necessary setup to generate parameters and keys."""
  # Setup boolean scheme parameters
  print('Setting up 128 bit security parameters, this may take upto 2 minutes')
  boolean_params = bool_params.get_params_for_128_bit_security()
  lwe_rng = bool_params.get_lwe_rng_for_128_bit_security(1)
  rlwe_rng = bool_params.get_rlwe_rng_for_128_bit_security(1)
  cks = jaxite_bool.ClientKeySet(boolean_params, lwe_rng, rlwe_rng)
  sks = jaxite_bool.ServerKeySet(
      cks,
      boolean_params,
      lwe_rng,
      rlwe_rng,
  )
  return (boolean_params, lwe_rng, cks, sks)


def setup_input_output(
    cks: jaxite_bool.ClientKeySet, lwe_rng: random_source.PseudorandomSource
) -> list[types.LweCiphertext]:
  """Perform setup for input and output ciphertexts to run benchmark."""
  cleartext: list[bool] = type_converters.str_to_cleartext(
      INPUT_STRING, static_len=STRING_SIZE
  )
  return [jaxite_bool.encrypt(x, cks, lwe_rng) for x in cleartext]


def main(argv: Sequence[str]) -> None:

  #   print('Setting up test parameters')
  #   (boolean_params, lwe_rng, cks, sks) = setup_test_params_keys()

  print('Setting up prod parameters')
  (boolean_params, lwe_rng, cks, sks) = setup_params_keys()

  ciphertext = setup_input_output(cks, lwe_rng)

  print('Jit compiling')
  result = redact_ssn(ciphertext, sks, boolean_params)

  print('Running benchmark')

  def timed_fn():
    result = redact_ssn(ciphertext, sks, boolean_params)
    for c in result:
      c.block_until_ready()

  timer = timeit.Timer(timed_fn)
  execution_time = timer.repeat(repeat=1, number=1)
  print('Redact ssn execution time: ', execution_time, 'seconds')

  # Decrypt
  result_cleartext = [jaxite_bool.decrypt(x, cks) for x in result]
  result = type_converters.cleartext_to_str(result_cleartext).strip()
  print('Result: --', result, '--')


if __name__ == '__main__':
  app.run(main)
