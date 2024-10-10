// RUN: heir-opt %s --verify-diagnostics 2>&1

func.func @test_uniform_distribution_verifier(%arg0: !random.prng) -> () {
  // expected-error@below {{Expected min less than max, found min = 1 and max = 0}}
  %1 = random.discrete_uniform_distribution %arg0 {range = [1, 0]} : (!random.prng) -> !random.distribution<distribution_type = uniform>
  return
}
