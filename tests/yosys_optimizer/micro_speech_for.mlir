// RUN: heir-opt --yosys-optimizer='abc-fast=true' %s | FileCheck %s

// CHECK: module
module {
    func.func @for_25_20_8(%98: i32, %99: i32, %100: i8) -> (i8) {
        // The only arith op we expect is arith.constant
        // CHECK-NOT: arith.{{^constant}}
        // CHECK: comb.truth_table
        %c1_i64 = arith.constant 1 : i64
        %c1073741824_i64 = arith.constant 1073741824 : i64
        %c0_i32 = arith.constant 0 : i32
        %c-1073741824_i64 = arith.constant -1073741824 : i64
        %c31_i32 = arith.constant 31 : i32
        %c-128_i32 = arith.constant -128 : i32
        %c127_i32 = arith.constant 127 : i32
        %101 = arith.extui %100 : i8 to i32
        %102 = arith.extsi %98 : i32 to i64
        %103 = arith.extsi %99 : i32 to i64
        %104 = arith.muli %102, %103 : i64
        %105 = arith.extui %100 : i8 to i64
        %106 = arith.shli %c1_i64, %105 : i64
        %107 = arith.shrui %106, %c1_i64 : i64
        %108 = arith.addi %104, %107 : i64
        %109 = arith.cmpi sge, %98, %c0_i32 : i32
        %110 = arith.select %109, %c1073741824_i64, %c-1073741824_i64 : i64
        %111 = arith.addi %110, %108 : i64
        %112 = arith.cmpi sgt, %101, %c31_i32 : i32
        %113 = arith.select %112, %111, %108 : i64
        %114 = arith.shrsi %113, %105 : i64
        %115 = arith.trunci %114 : i64 to i32
        %116 = arith.addi %115, %c-128_i32 : i32
        %117 = arith.cmpi slt, %116, %c-128_i32 : i32
        %118 = arith.select %117, %c-128_i32, %116 : i32
        %119 = arith.cmpi sgt, %116, %c127_i32 : i32
        %120 = arith.select %119, %c127_i32, %118 : i32
        %121 = arith.trunci %120 : i32 to i8
        // CHECK: return
        func.return %121 : i8
    }
}
