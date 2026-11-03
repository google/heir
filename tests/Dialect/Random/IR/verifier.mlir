// RUN: heir-opt %s --verify-diagnostics --split-input-file 2>&1

func.func @test_uniform_distribution_verifier(%arg0: !random.prng) -> () {
  // expected-error@below {{Expected min less than max, found min = 1 and max = 0}}
  %1 = random.discrete_uniform_distribution %arg0 {range = [1, 0]} : (!random.prng) -> !random.distribution<distribution_type = uniform>
  return
}

// -----

func.func @test_sample_modq_integer(%arg0: !random.prng) -> () {
  // expected-error@below {{result #0 must be mod_q-like}}
  %0 = random.sample_modq %arg0 : (!random.prng) -> tensor<8xi64>
  return
}
