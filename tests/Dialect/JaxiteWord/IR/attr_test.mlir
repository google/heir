// RUN: heir-opt %s | FileCheck %s

// CHECK: module attributes {jaxiteword.ckks_params = #jaxiteword.ckks_parameters<q_towers = [1, 2], p_towers = [3], r = 4, c = 5, dnum = 6, composite_degree = 7, batch = 8>}
module attributes {
  jaxiteword.ckks_params = #jaxiteword.ckks_parameters<
    q_towers = [1, 2],
    p_towers = [3],
    r = 4,
    c = 5,
    dnum = 6,
    composite_degree = 7,
    batch = 8
  >
} {
}
