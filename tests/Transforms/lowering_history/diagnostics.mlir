// RUN: not heir-opt %s --pass-pipeline='builtin.module(explain-lowering-history,record-lowering-history{stage=imported},record-lowering-history{stage=normalized},polynomial-approximation)' 2>&1 | FileCheck %s
// RUN: heir-opt %s --record-lowering-history=stage=serialized --mlir-print-debuginfo -o %t
// RUN: not heir-opt %t --explain-lowering-history --polynomial-approximation 2>&1 | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: not heir-opt %s --pass-pipeline='builtin.module(explain-lowering-history{max-notes=1},record-lowering-history{stage=imported},record-lowering-history{stage=normalized},polynomial-approximation)' 2>&1 | FileCheck %s --check-prefix=LIMIT
// RUN: not heir-opt %s --polynomial-approximation 2>&1 | FileCheck %s --check-prefix=OFF

// CHECK: error: {{.*}}domain_lower must be strictly less than domain_upper
// CHECK: note: lowering history: 'math.exp' observed at stage 'normalized'
// CHECK: note: lowering history: 'math.exp' observed at stage 'imported'
// ROUNDTRIP: error: {{.*}}domain_lower must be strictly less than domain_upper
// ROUNDTRIP: note: lowering history: 'math.exp' observed at stage 'serialized'
// LIMIT: note: lowering history: 'math.exp' observed at stage 'normalized'
// LIMIT: note: lowering history: additional checkpoints omitted
// LIMIT-NOT: observed at stage 'imported'
// OFF: error: {{.*}}domain_lower must be strictly less than domain_upper
// OFF-NOT: lowering history:
func.func @bad_domain(%x: f32 {secret.secret}) -> f32 {
  %y = math.exp %x {domain_lower = 1.0 : f64, domain_upper = 0.0 : f64} : f32 loc("model.py":42:1)
  return %y : f32
}
