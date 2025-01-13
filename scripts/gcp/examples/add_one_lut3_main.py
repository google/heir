import timeit

from jaxite.jaxite_bool import bool_params
from jaxite.jaxite_bool import jaxite_bool
from jaxite.jaxite_bool import type_converters
import add_one_lut3_lib


# Note: In real applications, a cryptographically secure seed needs to be
# used.
lwe_rng = bool_params.get_lwe_rng_for_128_bit_security(seed=1)
rlwe_rng = bool_params.get_rlwe_rng_for_128_bit_security(seed=1)
params = bool_params.get_params_for_128_bit_security()

cks = jaxite_bool.ClientKeySet(
    params,
    lwe_rng=lwe_rng,
    rlwe_rng=rlwe_rng,
)
print("Client keygen done")

sks = jaxite_bool.ServerKeySet(
    cks,
    params,
    lwe_rng=lwe_rng,
    rlwe_rng=rlwe_rng,
    bootstrap_callback=None,
)
print("Server keygen done.")

x = 5
cleartext_x = type_converters.u8_to_bit_slice(x)
ciphertext_x = [jaxite_bool.encrypt(z, cks, lwe_rng) for z in cleartext_x]

result_ciphertext = add_one_lut3_lib.add_one_lut3(
    ciphertext_x, sks, params
)

# Using Timeit
def timed_fn():
  result_ciphertext = add_one_lut3_lib.add_one_lut3(
      ciphertext_x, sks, params
  )
  for c in result_ciphertext:
    c.block_until_ready()

timer = timeit.Timer(timed_fn)
execution_time = timer.repeat(repeat=1, number=1)
print("Add one execution time: ", execution_time)

expected = x + 1
actual = type_converters.bit_slice_to_u8(
    [jaxite_bool.decrypt(z, cks) for z in result_ciphertext]
)
print(f"{actual=}, {expected=}")
