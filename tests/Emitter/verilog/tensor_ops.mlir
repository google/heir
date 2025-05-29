// RUN: heir-translate --emit-verilog %s | FileCheck %s

module {
  func.func @insert(%arg0: tensor<1xi1>, %arg1: i1, %arg2: i1) -> tensor<1xi1> {
    %c0 = arith.constant 0 : index
    %0 = arith.addi %arg1, %arg2 : i1
    %inserted = tensor.insert %0 into %arg0[%c0] : tensor<1xi1>
    return %inserted : tensor<1xi1>
  }
}

// CHECK: module insert(
// CHECK:  input wire [[arg1:.*]],
// CHECK:  input wire [[arg2:.*]],
// CHECK:  input wire [[arg3:.*]],
// CHECK:  output wire [[_out_0:.*]]
// CHECK: );
//
// CHECK:  assign [[v4:.*]] = [[arg2]] + [[arg3]];
// CHECK:  assign [[v5:.*]][0 == 0 ? 0 : 0 * 1 - 1 < 0 : 0] = [[arg1]][0 == 0 ? 0 : 0 * 1 - 1 < 0 : 0];
// CHECK:  assign [[v5]][0 + 1 * 0 : 1 * 0] = [[v4]];
// CHECK:  assign [[v5]][1 : 1 + 0 * 1] = [[arg1]][1 : 1 + 0 * 1];
// CHECK:  assign [[_out_0]] = [[v5]];
// CHECK: endmodule

// -----

module {
  func.func @extract(%arg0: tensor<1xi1>) -> i1 {
    %c0 = arith.constant 0 : index
    %extracted = tensor.extract %arg0[%c0] : tensor<1xi1>
    return %extracted : i1
  }
}

// CHECK: module extract(
// CHECK:  input wire [[arg1:.*]],
// CHECK:  output wire [[_out_0:.*]]
// CHECK: );
//
// CHECK:  assign [[v2:.*]] = [[arg1]][0 + 1 * 0 : 1 * 0];
// CHECK:  assign [[_out_0]] = [[v2]];
// CHECK: endmodule
