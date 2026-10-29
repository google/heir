// RUN: heir-opt --compare-to-sign-rewrite %s | FileCheck %s

// Regression test for https://github.com/google/heir/issues/3562
//
// Since sign(x) is -1 for x < 0, 0 for x == 0 and 1 for x > 0, the strict
// comparisons must be rewritten as
//   a < b -> 0.5 * (sign(b - a) + 1)
//   a > b -> 0.5 * (sign(a - b) + 1)

// CHECK: func @cmpf_olt
// CHECK-SAME: [[A:%.*]]: f32, [[B:%.*]]: f32
func.func @cmpf_olt(%a: f32, %b: f32) -> i1 {
  // CHECK: [[SUB:%.*]] = arith.subf [[B]], [[A]]
  // CHECK: [[SIGN:%.*]] = math_ext.sign [[SUB]]
  // CHECK-DAG: [[ONE:%.*]] = arith.constant 1.0
  // CHECK-DAG: [[HALF:%.*]] = arith.constant 5.0
  // CHECK: [[ADD:%.*]] = arith.addf [[SIGN]], [[ONE]]
  // CHECK: [[MUL:%.*]] = arith.mulf [[ADD]], [[HALF]]
  // CHECK: [[RES:%.*]] = arith.fptosi [[MUL]] : f32 to i1
  // CHECK-NOT: arith.cmpf
  %0 = arith.cmpf olt, %a, %b : f32
  // CHECK: return [[RES]] : i1
  return %0 : i1
}

// CHECK: func @cmpf_ult
// CHECK-SAME: [[A:%.*]]: f32, [[B:%.*]]: f32
func.func @cmpf_ult(%a: f32, %b: f32) -> i1 {
  // CHECK: [[SUB:%.*]] = arith.subf [[B]], [[A]]
  // CHECK: math_ext.sign [[SUB]]
  // CHECK-NOT: arith.cmpf
  %0 = arith.cmpf ult, %a, %b : f32
  return %0 : i1
}

// CHECK: func @cmpf_ogt
// CHECK-SAME: [[A:%.*]]: f32, [[B:%.*]]: f32
func.func @cmpf_ogt(%a: f32, %b: f32) -> i1 {
  // CHECK: [[SUB:%.*]] = arith.subf [[A]], [[B]]
  // CHECK: [[SIGN:%.*]] = math_ext.sign [[SUB]]
  // CHECK-DAG: [[ONE:%.*]] = arith.constant 1.0
  // CHECK-DAG: [[HALF:%.*]] = arith.constant 5.0
  // CHECK: [[ADD:%.*]] = arith.addf [[SIGN]], [[ONE]]
  // CHECK: [[MUL:%.*]] = arith.mulf [[ADD]], [[HALF]]
  // CHECK: [[RES:%.*]] = arith.fptosi [[MUL]] : f32 to i1
  // CHECK-NOT: arith.cmpf
  %0 = arith.cmpf ogt, %a, %b : f32
  // CHECK: return [[RES]] : i1
  return %0 : i1
}

// CHECK: func @cmpf_ugt
// CHECK-SAME: [[A:%.*]]: f32, [[B:%.*]]: f32
func.func @cmpf_ugt(%a: f32, %b: f32) -> i1 {
  // CHECK: [[SUB:%.*]] = arith.subf [[A]], [[B]]
  // CHECK: math_ext.sign [[SUB]]
  // CHECK-NOT: arith.cmpf
  %0 = arith.cmpf ugt, %a, %b : f32
  return %0 : i1
}

// Equality predicates are not handled by this pass.
// CHECK: func @cmpf_oeq
func.func @cmpf_oeq(%a: f32, %b: f32) -> i1 {
  // CHECK: arith.cmpf oeq
  // CHECK-NOT: math_ext.sign
  %0 = arith.cmpf oeq, %a, %b : f32
  return %0 : i1
}

// CHECK: func @cmpi_slt
// CHECK-SAME: [[A:%.*]]: i32, [[B:%.*]]: i32
func.func @cmpi_slt(%a: i32, %b: i32) -> i1 {
  // CHECK-DAG: [[FA:%.*]] = arith.sitofp [[A]] : i32 to f32
  // CHECK-DAG: [[FB:%.*]] = arith.sitofp [[B]] : i32 to f32
  // CHECK: [[SUB:%.*]] = arith.subf [[FB]], [[FA]]
  // CHECK: [[SIGN:%.*]] = math_ext.sign [[SUB]]
  // CHECK-DAG: [[ONE:%.*]] = arith.constant 1.0
  // CHECK-DAG: [[HALF:%.*]] = arith.constant 5.0
  // CHECK: [[ADD:%.*]] = arith.addf [[SIGN]], [[ONE]]
  // CHECK: [[MUL:%.*]] = arith.mulf [[ADD]], [[HALF]]
  // CHECK: [[RES:%.*]] = arith.fptosi [[MUL]] : f32 to i1
  // CHECK-NOT: arith.cmpi
  %0 = arith.cmpi slt, %a, %b : i32
  // CHECK: return [[RES]] : i1
  return %0 : i1
}

// CHECK: func @cmpi_ult
// CHECK-SAME: [[A:%.*]]: i32, [[B:%.*]]: i32
func.func @cmpi_ult(%a: i32, %b: i32) -> i1 {
  // CHECK-DAG: [[FA:%.*]] = arith.sitofp [[A]] : i32 to f32
  // CHECK-DAG: [[FB:%.*]] = arith.sitofp [[B]] : i32 to f32
  // CHECK: [[SUB:%.*]] = arith.subf [[FB]], [[FA]]
  // CHECK: math_ext.sign [[SUB]]
  // CHECK-NOT: arith.cmpi
  %0 = arith.cmpi ult, %a, %b : i32
  return %0 : i1
}

// CHECK: func @cmpi_sgt
// CHECK-SAME: [[A:%.*]]: i32, [[B:%.*]]: i32
func.func @cmpi_sgt(%a: i32, %b: i32) -> i1 {
  // CHECK-DAG: [[FA:%.*]] = arith.sitofp [[A]] : i32 to f32
  // CHECK-DAG: [[FB:%.*]] = arith.sitofp [[B]] : i32 to f32
  // CHECK: [[SUB:%.*]] = arith.subf [[FA]], [[FB]]
  // CHECK: [[SIGN:%.*]] = math_ext.sign [[SUB]]
  // CHECK-DAG: [[ONE:%.*]] = arith.constant 1.0
  // CHECK-DAG: [[HALF:%.*]] = arith.constant 5.0
  // CHECK: [[ADD:%.*]] = arith.addf [[SIGN]], [[ONE]]
  // CHECK: [[MUL:%.*]] = arith.mulf [[ADD]], [[HALF]]
  // CHECK: [[RES:%.*]] = arith.fptosi [[MUL]] : f32 to i1
  // CHECK-NOT: arith.cmpi
  %0 = arith.cmpi sgt, %a, %b : i32
  // CHECK: return [[RES]] : i1
  return %0 : i1
}

// CHECK: func @cmpi_ugt
// CHECK-SAME: [[A:%.*]]: i32, [[B:%.*]]: i32
func.func @cmpi_ugt(%a: i32, %b: i32) -> i1 {
  // CHECK-DAG: [[FA:%.*]] = arith.sitofp [[A]] : i32 to f32
  // CHECK-DAG: [[FB:%.*]] = arith.sitofp [[B]] : i32 to f32
  // CHECK: [[SUB:%.*]] = arith.subf [[FA]], [[FB]]
  // CHECK: math_ext.sign [[SUB]]
  // CHECK-NOT: arith.cmpi
  %0 = arith.cmpi ugt, %a, %b : i32
  return %0 : i1
}

// CHECK: func @cmpi_index_slt
// CHECK-SAME: [[A:%.*]]: index, [[B:%.*]]: index
func.func @cmpi_index_slt(%a: index, %b: index) -> i1 {
  // CHECK-DAG: [[IA:%.*]] = arith.index_cast [[A]] : index to i32
  // CHECK-DAG: [[IB:%.*]] = arith.index_cast [[B]] : index to i32
  // CHECK-DAG: [[FA:%.*]] = arith.sitofp [[IA]] : i32 to f32
  // CHECK-DAG: [[FB:%.*]] = arith.sitofp [[IB]] : i32 to f32
  // CHECK: [[SUB:%.*]] = arith.subf [[FB]], [[FA]]
  // CHECK: math_ext.sign [[SUB]]
  // CHECK-NOT: arith.cmpi
  %0 = arith.cmpi slt, %a, %b : index
  return %0 : i1
}

// Equality predicates are not handled by this pass.
// CHECK: func @cmpi_eq
func.func @cmpi_eq(%a: i32, %b: i32) -> i1 {
  // CHECK: arith.cmpi eq
  // CHECK-NOT: math_ext.sign
  %0 = arith.cmpi eq, %a, %b : i32
  return %0 : i1
}
