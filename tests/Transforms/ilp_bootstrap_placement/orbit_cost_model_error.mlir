// RUN: not heir-opt --ilp-bootstrap-placement="orbit-cost-model=%S/does_not_exist.json" %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not heir-opt --ilp-bootstrap-placement="orbit-cost-model=%S/orbit_bad_cost_model.json" %s 2>&1 | FileCheck %s --check-prefix=MALFORMED

// MISSING: failed to load Orbit cost model
// MALFORMED: failed to load Orbit cost model

func.func @orbit_cost_model_error(
    %arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>) {
  ^body(%input0: tensor<8xf32>):
    %out = arith.addf %input0, %input0 : tensor<8xf32>
    secret.yield %out : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
