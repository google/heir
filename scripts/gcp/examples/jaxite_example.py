"""Example script showing how to use Jaxite."""

import timeit

from jaxite.jaxite_bool import jaxite_bool


bool_params = jaxite_bool.bool_params

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

ct_true = jaxite_bool.encrypt(True, cks, lwe_rng)
ct_false = jaxite_bool.encrypt(False, cks, lwe_rng)

# Calling function once before timing it so compile-time doesn't get
# included in timing metircs.
and_gate = jaxite_bool.and_(ct_false, ct_true, sks, params)

# Using Timeit
def timed_fn():
  and_gate = jaxite_bool.and_(ct_false, ct_true, sks, params)
  and_gate.block_until_ready()
timer = timeit.Timer(timed_fn)
execution_time = timer.repeat(repeat=1, number=1)
print("And gate execution time: ", execution_time)

actual = jaxite_bool.decrypt(and_gate, cks)
expected = False
print(f"{actual=}, {expected=}")
