// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 scale-waterline=51 scale-factor-bits=51" %s | FileCheck %s --check-prefix=CHECK-FLAT
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 scale-waterline=51 scale-factor-bits=51 orbit-cost-model=%S/orbit_cost_model.json" %s | FileCheck %s --check-prefix=CHECK-LEVEL

// The circuit is mul -> adds -> mul, which needs exactly one rescale on each
// mul result to reach the annotated output state. The rescale between the two
// muls can be placed before or after the add chain without changing the
// rescale count, so the placement is decided by level preferences alone.
//
// With flat costs, the high-level tie-breaker keeps the adds at level 3 and
// rescales after the add chain. With Orbit-style level-dependent costs, adds
// are cheaper at lower levels, so the rescale moves directly after the first
// mul and the adds run at level 2.

// CHECK-FLAT: func.func @level_dependent_rescale_timing
// CHECK-FLAT: arith.mulf %input0, %input0
// CHECK-FLAT-NEXT: mgmt.relinearize
// CHECK-FLAT-NEXT: arith.addf
// CHECK-FLAT: mgmt.modreduce
// CHECK-FLAT: arith.mulf
// CHECK-FLAT-NEXT: mgmt.relinearize
// CHECK-FLAT-NEXT: mgmt.modreduce
// CHECK-FLAT-NOT: mgmt.bootstrap

// CHECK-LEVEL: func.func @level_dependent_rescale_timing
// CHECK-LEVEL: arith.mulf %input0, %input0
// CHECK-LEVEL-NEXT: mgmt.relinearize
// CHECK-LEVEL-NEXT: mgmt.modreduce
// CHECK-LEVEL-NEXT: arith.addf
// CHECK-LEVEL: arith.mulf
// CHECK-LEVEL-NEXT: mgmt.relinearize
// CHECK-LEVEL-NEXT: mgmt.modreduce
// CHECK-LEVEL-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  func.func @level_dependent_rescale_timing(
      %arg0: !ct_ty)
      -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 51>}) {
    %0 = secret.generic(%arg0: !ct_ty) {
    ^body(%input0: !pt_ty):
      %m = arith.mulf %input0, %input0 : !pt_ty
      %a1 = arith.addf %m, %m : !pt_ty
      %a2 = arith.addf %a1, %a1 : !pt_ty
      %a3 = arith.addf %a2, %a2 : !pt_ty
      %mm = arith.mulf %a3, %a3 : !pt_ty
      secret.yield %mm : !pt_ty
    } -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 51>})
    return %0 : !ct_ty
  }
}
