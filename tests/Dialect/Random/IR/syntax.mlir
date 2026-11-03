// This test only checks that the syntax is correct.

// RUN: heir-opt %s

func.func @test_random(%arg0: index) -> () {
  %0 = random.init_prng %arg0 <num_bits = 32>: (index) -> !random.prng
  %1 = random.discrete_gaussian_distribution %0 <mean = 10, stddev = 10> : (!random.prng) -> !random.distribution<distribution_type = gaussian>
  %2 = random.sample %1 : (!random.distribution<distribution_type = gaussian>) -> i32

  %4 = random.init_prng %arg0 <num_bits = 32>: (index) -> !random.prng
  %5 = random.discrete_uniform_distribution %4 {range = [-1, 2]} : (!random.prng) -> !random.distribution<distribution_type = uniform>
  %6 = random.sample %5 : (!random.distribution<distribution_type = uniform>) -> tensor<10xi32>
  return
}

!Z1095233372161 = !mod_arith.int<1095233372161 : i64>
!Z1032955396097 = !mod_arith.int<1032955396097 : i64>
!rns = !rns.rns<!Z1095233372161, !Z1032955396097>

func.func @test_sample_modq(%arg0: index) -> () {
  %0 = random.init_prng %arg0 <num_bits = 32>: (index) -> !random.prng
  %1 = random.sample_modq %0 : (!random.prng) -> !Z1095233372161
  %2 = random.sample_modq %0 : (!random.prng) -> tensor<8x!Z1095233372161>
  %3 = random.sample_modq %0 : (!random.prng) -> !rns
  %4 = random.sample_modq %0 : (!random.prng) -> tensor<8x!rns>
  return
}
