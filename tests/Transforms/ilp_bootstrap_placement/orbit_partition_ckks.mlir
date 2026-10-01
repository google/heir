// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 scale-waterline=51 scale-factor-bits=51 orbit-cost-model=%S/orbit_cost_model.json partition-min-size=1" %s | FileCheck %s

// The CKKS variant of partition stitching: boundary states are (level, scale)
// pairs, with the boundary scale realized by the solver under the
// output-scale pressure term rather than enumerated, so the stitched rescale
// positions may differ from the single-partition optimum on circuits this
// small (Orbit has the same property). What must hold: the annotated output
// state is reached with exactly two rescales and no bootstrap.

// CHECK: func.func @partitioned_level_dependent_rescale
// CHECK-COUNT-2: mgmt.modreduce
// CHECK-NOT: mgmt.modreduce
// CHECK-NOT: mgmt.bootstrap

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

module attributes {scheme.ckks} {
  func.func @partitioned_level_dependent_rescale(
      %arg0: !ct_ty)
      -> (!ct_ty {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 51>}) {
    %0 = secret.generic(%arg0: !ct_ty) {
    ^body(%input0: tensor<8xf32>):
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
