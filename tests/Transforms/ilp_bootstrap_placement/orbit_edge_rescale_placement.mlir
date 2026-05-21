// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s --check-prefix=CHECK-RESCALE
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 bootstrap-cost=1 rescale-cost=100000000" %s | FileCheck %s --check-prefix=CHECK-BTS

// The Orbit-style ILP can reduce a single incoming edge instead of forcing the
// producer SSA value itself to live at the consumer level. Here %input0 remains
// fresh for other uses, while the add consumes a level-reduced copy. The
// multiplication uses a plaintext constant so rescale is feasible with the
// default Orbit-style Sw=40 and scale-factor-bits=51 model.

// CHECK-RESCALE: func.func @orbit_edge_rescale_placement
// CHECK-RESCALE: %[[PLAIN:.*]] = arith.constant
// CHECK-RESCALE: secret.generic
// CHECK-RESCALE: arith.mulf %input0, %[[PLAIN]]
// CHECK-RESCALE-NEXT: mgmt.modreduce
// CHECK-RESCALE-NOT: mgmt.bootstrap
// CHECK-RESCALE: %[[REDUCED:.*]] = mgmt.level_reduce %input0
// CHECK-RESCALE: arith.addf %[[REDUCED]],
// CHECK-RESCALE-NOT: mgmt.bootstrap

// CHECK-BTS: func.func @orbit_edge_rescale_placement
// CHECK-BTS: %[[PLAIN:.*]] = arith.constant
// CHECK-BTS: secret.generic
// CHECK-BTS: arith.mulf %input0, %[[PLAIN]]
// CHECK-BTS-NOT: mgmt.level_reduce %input0
// CHECK-BTS: %[[BOOTSTRAPPED:.*]] = mgmt.bootstrap
// CHECK-BTS: arith.addf %input0, %[[BOOTSTRAPPED]]

func.func @orbit_edge_rescale_placement(
    %arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(
      %arg0: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>):
    %plain = arith.constant dense<2.0> : tensor<8xf32>
    %l1 = arith.mulf %input0, %plain : tensor<8xf32>
    %out = arith.addf %input0, %l1 : tensor<8xf32>
    secret.yield %out : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
