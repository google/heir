// This test only checks that the syntax is correct.

// RUN: heir-opt %s

func.func @test_random(%arg0: i32) -> () {
  %0 = random.init_prng %arg0 {num_bits = 32}: (i32) -> !random.prng
  %1 = random.discrete_gaussian_distribution %0 {mean=10, stddev=10} : (!random.prng) -> !random.distribution<distribution_type = gaussian>
  %2 = random.sample %1 : (!random.distribution<distribution_type = gaussian>) -> i32

  %4 = random.init_prng %arg0 {num_bits = 32}: (i32) -> !random.prng
  %5 = random.discrete_uniform_distribution %4 {range = [-1, 2]} : (!random.prng) -> !random.distribution<distribution_type = uniform>
  %6 = random.sample %5 : (!random.distribution<distribution_type = uniform>) -> tensor<10xi32>
  return
}
