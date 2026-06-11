// RUN: not heir-opt --ilp-bootstrap-placement="orbit-cost-model=%S/does_not_exist.json" %s 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: not heir-opt --ilp-bootstrap-placement="orbit-cost-model=%S/orbit_bad_cost_model.json" %s 2>&1 | FileCheck %s --check-prefix=MALFORMED

// MISSING: failed to load Orbit cost model
// MALFORMED: failed to load Orbit cost model

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @orbit_cost_model_error(%arg0: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%input0: !pt_ty):
    %out = arith.addf %input0, %input0 : !pt_ty
    secret.yield %out : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
