// RUN: heir-opt --secretize --convert-secret-for-to-static-for=convert-all-scf-for=true %s | FileCheck %s
func.func @foo(%arg0: f64 {secret.secret}, %arg1: f64 {secret.secret}) -> f64 {
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%cst = arith.constant 1.000000e+00 : f64
%0 = arith.fptosi %arg1 : f64 to i32
%1 = arith.index_cast %0 : i32 to index
// CHECK-NOT: scf.for
// CHECK: affine.for
// CHECK: affine.for
%2 = scf.for %arg2 = %c0 to %1 step %c1 iter_args(%arg3 = %cst) -> (f64) {
    %loop = affine.for %i = 0 to 10 iter_args(%x = %arg3) -> (f64) {
        %add = arith.addf %x, %arg0 : f64
        affine.yield %add : f64
    }
    scf.yield %loop : f64
} {lower = 0 : i64, upper = 6 : i64}
return %2 : f64
}
