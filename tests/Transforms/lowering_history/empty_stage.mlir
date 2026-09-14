// RUN: heir-opt %s --record-lowering-history --verify-diagnostics
// expected-error@+1 {{record-lowering-history requires a nonempty stage}}
module {}
