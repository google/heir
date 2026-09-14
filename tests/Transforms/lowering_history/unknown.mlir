// RUN: not heir-opt %s --pass-pipeline='builtin.module(explain-lowering-history,explain-lowering-history,record-lowering-history{stage=unknown-source},polynomial-approximation)' 2>&1 | FileCheck %s
// CHECK: error: {{.*}}domain_lower must be strictly less than domain_upper
// CHECK: note: lowering history: 'math.exp' observed at stage 'unknown-source'
// CHECK-NOT: lowering history:
func.func @unknown_source(%x: f32 {secret.secret}) -> f32 {
  %y = math.exp %x {domain_lower = 1.0 : f64, domain_upper = 0.0 : f64} : f32 loc(unknown)
  return %y : f32
}
