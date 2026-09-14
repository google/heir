// RUN: heir-opt %s --pass-pipeline='builtin.module(record-lowering-history{stage=input},activation-canonicalizations,polynomial-approximation)' --mlir-print-debuginfo | FileCheck %s
// RUN: heir-opt %s --activation-canonicalizations --polynomial-approximation --mlir-print-debuginfo | FileCheck %s --check-prefix=OFF
// CHECK: polynomial.eval
// CHECK-DAG: heir.lowering_pass = "activation-canonicalizations", heir.lowering_result = "arith.maximumf"
// CHECK-DAG: heir.lowering_pass = "polynomial-approximation", heir.lowering_result = "polynomial.eval"
// CHECK-DAG: loc("arith.select"(
// CHECK-DAG: loc("arith.maximumf"(
// OFF: polynomial.eval
// OFF-NOT: heir.lowering_
func.func @relu(%x: f32 {secret.secret}) -> f32 {
  %zero = arith.constant 0.0 : f32
  %cmp = arith.cmpf ugt, %x, %zero : f32
  %y = arith.select %cmp, %x, %zero : f32 loc("relu"("model.py":11:1))
  return %y : f32
}
