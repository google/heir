// This tests functional correctness through heir-translate for ctlz.
//
// RUN: heir-translate --emit-verilog %s 2>&1 > %t1
// RUN: run_verilog \
// RUN:  --verilog_module %t1 \
// RUN:  --input='arg1=0' \
// RUN:  --input='arg1=1' \
// RUN:  --input='arg1=65535' \
// RUN:  --input='arg1=4294967295' \
// RUN:  > %t
// RUN: FileCheck %s < %t

// CHECK: b'\x20'
// CHECK-NEXT: b'\x1f'
// CHECK-NEXT: b'\x10'
// CHECK-NEXT: b'\x00'

module {
  func.func @main(%arg0: i32) -> i32 {
    %1 = math.ctlz %arg0 : i32
    return %1 : i32
  }
}
