// RUN: heir-opt --ilp-bootstrap-placement=bootstrap-waterline=3 %s | FileCheck %s

// Orbit-style node rescale can reduce a producer once when all outgoing uses
// need the lower level. The shared add result is reduced once before its
// consumers, rather than independently at each consumer. The multiplication
// uses a plaintext constant so rescale is feasible with the default Orbit-style
// Sw=40 and scale-factor-bits=51 model.

// CHECK: func.func @orbit_output_rescale_placement
// CHECK: %[[PLAIN:.*]] = arith.constant
// CHECK: %[[SHARED:.*]] = arith.addf %input0, %input1
// CHECK-NEXT: %[[REDUCED:.*]] = mgmt.level_reduce %[[SHARED]]
// CHECK: arith.mulf %input0, %[[PLAIN]]
// CHECK: arith.addf %[[REDUCED]],
// CHECK: arith.subf %[[REDUCED]],
// CHECK-NOT: mgmt.bootstrap

func.func @orbit_output_rescale_placement(
    %arg0: !secret.secret<tensor<8xf32>>,
    %arg1: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(
      %arg0: !secret.secret<tensor<8xf32>>,
      %arg1: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>,
        %input1: tensor<8xf32>):
    %shared = arith.addf %input0, %input1 : tensor<8xf32>
    %plain = arith.constant dense<2.0> : tensor<8xf32>
    %l1 = arith.mulf %input0, %plain : tensor<8xf32>
    %use0 = arith.addf %shared, %l1 : tensor<8xf32>
    %use1 = arith.subf %shared, %l1 : tensor<8xf32>
    %out = arith.addf %use0, %use1 : tensor<8xf32>
    secret.yield %out : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
