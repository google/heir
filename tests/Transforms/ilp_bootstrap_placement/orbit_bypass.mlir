// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=5 bypass-depth-threshold=3" %s | FileCheck %s --check-prefix=BYPASS --implicit-check-not=mgmt.bootstrap
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=3 bypass-depth-threshold=3" %s | FileCheck %s --check-prefix=BOOT
// RUN: heir-opt --ilp-bootstrap-placement="bootstrap-waterline=5 bypass-depth-threshold=0" %s | FileCheck %s --check-prefix=OFF --implicit-check-not=mgmt.bootstrap

// A residual: a multiplicative-depth-4 multiply chain (main) and an identity
// skip of the input (bypass) rejoined by an addition. The input stays live
// across the whole chain, so the body has no single-input single-output cut.
// With bypass enabled the residual is solved by splitting the deep main path
// from the shallow skip instead of as one monolithic ILP; below the depth
// threshold it falls back to the monolithic solve. Both are feasible.

// Waterline 5 covers the depth-4 chain, so no bootstrap is needed.
// BYPASS-LABEL: func.func @residual
// BYPASS-COUNT-4: arith.mulf
// BYPASS: arith.addf
// BYPASS: secret.yield

// Waterline 3 overflows the chain, so the main path bootstraps once.
// BOOT-LABEL: func.func @residual
// BOOT-COUNT-1: mgmt.bootstrap
// BOOT-NOT: mgmt.bootstrap

// With the threshold at 0 bypass is disabled; the monolithic solve is still
// feasible and needs no bootstrap at waterline 5.
// OFF-LABEL: func.func @residual
// OFF: secret.yield

!pt_ty = tensor<8xf32>
!ct_ty = !secret.secret<!pt_ty>

func.func @residual(%arg0: !ct_ty) -> !ct_ty {
  %0 = secret.generic(%arg0: !ct_ty) {
  ^body(%in: !pt_ty):
    %m1 = arith.mulf %in, %in : !pt_ty
    %m2 = arith.mulf %m1, %m1 : !pt_ty
    %m3 = arith.mulf %m2, %m2 : !pt_ty
    %m4 = arith.mulf %m3, %m3 : !pt_ty
    %out = arith.addf %m4, %in : !pt_ty
    secret.yield %out : !pt_ty
  } -> !ct_ty
  return %0 : !ct_ty
}
