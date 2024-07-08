// RUN: heir-opt --convert-elementwise-to-affine %s | FileCheck --enable-var-scope %s

!poly = !polynomial.polynomial<ring=<coefficientType = i32, coefficientModulus = 33538049 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**1024>>>

// CHECK-LABEL:  @test_elementwise
// CHECK: {{.*}} -> [[T:tensor<2x!polynomial.*33538049.*]] {
func.func @test_elementwise(%arg0: tensor<2x!poly>, %arg1: tensor<2x!poly>) ->  tensor<2x!poly> {
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x!polynomial.*33538049.*]]
  // CHECK: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK: [[A:%.+]] = tensor.extract %arg0[[[I]]] : [[T]]
    // CHECK: [[B:%.+]] = tensor.extract %arg1[[[I]]] : [[T]]
    // CHECK: [[S:%.+]] = polynomial.add [[A]], [[B]] : [[P:!polynomial.*33538049.*]]
    // CHECK-NOT: polynomial.add{{.*}} : [[T]]
    // CHECK: [[R:%.+]] = tensor.insert [[S]] into [[T0]][[[I]]] : [[T]]
    // CHECK: affine.yield [[R]] : [[T]]
  %0 = polynomial.add %arg0, %arg1 : tensor<2x!poly>
  // CHECK: return [[LOOP]] : [[T]]
  return %0 :  tensor<2x!poly>
}

// CHECK-LABEL:  @test_partially_elementwise
// CHECK: ([[ARG0:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]], [[ARG1:%.+]]: i32) -> [[T]] {
func.func @test_partially_elementwise(%arg0: tensor<2x!poly>, %arg1: i32) ->  tensor<2x!poly> {
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x!polynomial.*33538049.*]]
  // CHECK: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK: [[A:%.+]] = tensor.extract [[ARG0]][[[I]]] : [[T]]
    // CHECK: [[S:%.+]] = polynomial.mul_scalar [[A]], [[ARG1]] : [[P:!polynomial.*33538049.*]], i32
    // CHECK-NOT: polynomial.mul_scalar{{.*}} : [[T]], i32
    // CHECK: [[R:%.+]] = tensor.insert [[S]] into [[T0]][[[I]]] : [[T]]
    // CHECK: affine.yield [[R]] : [[T]]
  %0 = polynomial.mul_scalar %arg0, %arg1 : tensor<2x!poly>, i32
  // CHECK: return [[LOOP]] : [[T]]
  return %0 :  tensor<2x!poly>
}

// CHECK-LABEL:  @test_elementwise_multidim
// CHECK: {{.*}} -> [[T:tensor<2x3x!polynomial.*33538049.*]] {
func.func @test_elementwise_multidim(%arg0: tensor<2x3x!poly>, %arg1: tensor<2x3x!poly>) ->  tensor<2x3x!poly> {
  %0 = polynomial.add %arg0, %arg1 : tensor<2x3x!poly>
  return %0 :  tensor<2x3x!poly>
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x3x!polynomial.*33538049.*]]
  // CHECK: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK: [[INNERLOOP:%.+]] = affine.for [[J:%.+]] = 0 to 3 iter_args([[T1:%.+]] = [[T0]]) -> ([[T]]) {
      // CHECK: [[A:%.+]] = tensor.extract %arg0[[[I]], [[J]]] : [[T]]
      // CHECK: [[B:%.+]] = tensor.extract %arg1[[[I]], [[J]]] : [[T]]
      // CHECK: [[S:%.+]] = polynomial.add [[A]], [[B]] : [[P:!polynomial.*33538049.*]]
      // CHECK-NOT: polynomial.add{{.*}} : [[T]]
      // CHECK: [[R:%.+]] = tensor.insert [[S]] into [[T1]][[[I]], [[J]]] : [[T]]
      // CHECK: affine.yield [[R]] : [[T]]
    // CHECK: affine.yield [[INNERLOOP]] : [[T]]
  // CHECK: return [[LOOP]] : [[T]]
}
