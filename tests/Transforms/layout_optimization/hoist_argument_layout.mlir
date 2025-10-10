// RUN: heir-opt --layout-optimization --canonicalize %s -split-input-file | FileCheck %s

// Cyclic repetition layout
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
// zero-padded layout
#layout1 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : i0 = slot and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">

// CHECK: #[[layout:.*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func.func @simple_add
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]},
  // CHECK-SAME:  %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]})
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: arith.addi %[[input0:.*]], %[[input1:.*]] {tensor_ext.layout = #[[layout]]} : tensor<32xi16>
  // CHECK: return
  func.func @simple_add(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = #layout, to_layout = #layout} : tensor<32xi16>
      %2 = arith.addi %1, %input1 {tensor_ext.layout = #layout} : tensor<32xi16>
      secret.yield %2 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<32xi16>>
  }
}

// -----

// Cyclic repetition layout
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
// zero-padded layout
#layout1 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : i0 = slot and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">

// CHECK-DAG: #[[layout:.*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
// CHECK-DAG: #[[layout1:.*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : i0 = slot and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func.func @different_conversions
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout1]]},
  // CHECK-SAME:  %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]})
  // CHECK: secret.generic
  // CHECK-SAME %[[arg0]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout1]]},
  // CHECK-COUNT-1: tensor_ext.convert_layout
  // CHECK: secret.generic
  // CHECK-SAME %[[arg1]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout1]]}
  // CHECK: return
  func.func @different_conversions(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}, !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = #layout, to_layout = #layout} : tensor<32xi16>
      %2 = arith.addi %1, %input1 {tensor_ext.layout = #layout} : tensor<32xi16>
      secret.yield %2 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout})
    %1 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<32xi16>):
      %2 = arith.addi %input0, %input0 {tensor_ext.layout = #layout1} : tensor<32xi16>
      secret.yield %2 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1})
    return %0, %1 : !secret.secret<tensor<32xi16>>, !secret.secret<tensor<32xi16>>
  }
}

// -----

// Cyclic repetition layout
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
// zero-padded layout
#layout1 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : i0 = slot and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">

// CHECK-DAG: #[[layout:.*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func.func @same_conversions
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]},
  // CHECK-SAME:  %[[arg1:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]})
  // CHECK: secret.generic
  // CHECK-SAME %[[arg0]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]},
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: secret.generic
  // CHECK-SAME %[[arg1]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]}
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: return
  func.func @same_conversions(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}, !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}) {
    %0 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = #layout, to_layout = #layout} : tensor<32xi16>
      %2 = arith.addi %1, %input1 {tensor_ext.layout = #layout} : tensor<32xi16>
      secret.yield %2 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout})
    %1 = secret.generic(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    ^body(%input0: tensor<32xi16>, %input1: tensor<32xi16>):
      %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = #layout, to_layout = #layout} : tensor<32xi16>
      %2 = arith.addi %1, %input1 {tensor_ext.layout = #layout} : tensor<32xi16>
      secret.yield %2 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout})
    return %0, %1 : !secret.secret<tensor<32xi16>>, !secret.secret<tensor<32xi16>>
  }
}

// -----

// Cyclic repetition layout
#layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">

// CHECK-DAG: #[[layout:.*]] = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 32 = 0 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
module {
  // CHECK: func.func @return
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #[[layout]]})
  // CHECK: return %[[arg0:.*]] : !secret.secret<tensor<32xi16>>
  func.func @return(%arg0: !secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout}) {
    return %arg0 : !secret.secret<tensor<32xi16>>
  }
}
