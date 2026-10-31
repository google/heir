// RUN: heir-opt --mod-arith-to-arith %s | FileCheck %s

!Zp32 = !mod_arith.int<974258177 : i32>
!Zp64 = !mod_arith.int<1125899908022273 : i64>

// CHECK: func.func @test_lower_mont_mul_i32
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: %[[LO:.*]], %[[HI:.*]] = arith.mulsi_extended %arg0, %arg1 : i32
// CHECK: %[[QINV:.*]] = arith.constant -974258175 : i32
// CHECK: %[[M:.*]] = arith.muli %[[LO]], %[[QINV]] : i32
// CHECK: %[[Q:.*]] = arith.constant 974258177 : i32
// CHECK: %{{.*}}, %[[T:.*]] = arith.mulsi_extended %[[M]], %[[Q]] : i32
// CHECK: %[[R:.*]] = arith.subi %[[HI]], %[[T]] : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[R]], %[[ZERO]] : i32
// CHECK: %[[R_PLUS_Q:.*]] = arith.addi %[[R]], %[[Q]] : i32
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[R_PLUS_Q]], %[[R]] : i32
// CHECK: return %[[RES]] : i32
func.func @test_lower_mont_mul_i32(%a: !Zp32, %b: !Zp32) -> !Zp32 {
  %0 = mod_arith.mont_mul %a, %b : !Zp32
  return %0 : !Zp32
}

// CHECK: func.func @test_lower_mont_mul_lazy_i32
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK-NOT: arith.select
// CHECK: %[[LO:.*]], %[[HI:.*]] = arith.mulsi_extended %arg0, %arg1 : i32
// CHECK: %[[QINV:.*]] = arith.constant -974258175 : i32
// CHECK: %[[M:.*]] = arith.muli %[[LO]], %[[QINV]] : i32
// CHECK: %[[Q:.*]] = arith.constant 974258177 : i32
// CHECK: %{{.*}}, %[[T:.*]] = arith.mulsi_extended %[[M]], %[[Q]] : i32
// CHECK: %[[R:.*]] = arith.subi %[[HI]], %[[T]] : i32
// CHECK: return %[[R]] : i32
func.func @test_lower_mont_mul_lazy_i32(%a: !Zp32, %b: !Zp32) -> !Zp32 {
  %0 = mod_arith.mont_mul %a, %b lazy : !Zp32
  return %0 : !Zp32
}

// CHECK: func.func @test_lower_mont_reduce_i32
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: %[[LO:.*]] = arith.trunci %arg0 : i64 to i32
// CHECK: %[[C32:.*]] = arith.constant 32 : i64
// CHECK: %[[SHR:.*]] = arith.shrsi %arg0, %[[C32]] : i64
// CHECK: %[[HI:.*]] = arith.trunci %[[SHR]] : i64 to i32
// CHECK: %[[QINV:.*]] = arith.constant -974258175 : i32
// CHECK: %[[M:.*]] = arith.muli %[[LO]], %[[QINV]] : i32
// CHECK: %[[Q:.*]] = arith.constant 974258177 : i32
// CHECK: %{{.*}}, %[[T:.*]] = arith.mulsi_extended %[[M]], %[[Q]] : i32
// CHECK: %[[R:.*]] = arith.subi %[[HI]], %[[T]] : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[R]], %[[ZERO]] : i32
// CHECK: %[[R_PLUS_Q:.*]] = arith.addi %[[R]], %[[Q]] : i32
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[R_PLUS_Q]], %[[R]] : i32
// CHECK: return %[[RES]] : i32
func.func @test_lower_mont_reduce_i32(%w: i64) -> !Zp32 {
  %0 = mod_arith.mont_reduce %w : i64 -> !Zp32
  return %0 : !Zp32
}

// CHECK: func.func @test_lower_to_mont_i32
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: %[[R2:.*]] = arith.constant 850126382 : i32
// CHECK: %{{.*}}, %{{.*}} = arith.mulsi_extended %arg0, %[[R2]] : i32
func.func @test_lower_to_mont_i32(%a: !Zp32) -> !Zp32 {
  %0 = mod_arith.to_mont %a : !Zp32
  return %0 : !Zp32
}

// CHECK: func.func @test_lower_from_mont_i32
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[QINV:.*]] = arith.constant -974258175 : i32
// CHECK: %[[M:.*]] = arith.muli %arg0, %[[QINV]] : i32
// CHECK: %[[Q:.*]] = arith.constant 974258177 : i32
// CHECK: %{{.*}}, %[[T:.*]] = arith.mulsi_extended %[[M]], %[[Q]] : i32
// CHECK: %[[R:.*]] = arith.subi %[[ZERO]], %[[T]] : i32
// CHECK: %[[CMP:.*]] = arith.cmpi slt, %[[R]], %{{.*}} : i32
// CHECK: %[[R_PLUS_Q:.*]] = arith.addi %[[R]], %[[Q]] : i32
// CHECK: %[[RES:.*]] = arith.select %[[CMP]], %[[R_PLUS_Q]], %[[R]] : i32
// CHECK: return %[[RES]] : i32
func.func @test_lower_from_mont_i32(%a: !Zp32) -> !Zp32 {
  %0 = mod_arith.from_mont %a : !Zp32
  return %0 : !Zp32
}

// CHECK: func.func @test_lower_mont_mul_i64
// CHECK-NOT: arith.remui
// CHECK-NOT: arith.remsi
// CHECK: arith.mulsi_extended %arg0, %arg1 : i64
func.func @test_lower_mont_mul_i64(%a: !Zp64, %b: !Zp64) -> !Zp64 {
  %0 = mod_arith.mont_mul %a, %b : !Zp64
  return %0 : !Zp64
}
