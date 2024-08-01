// THE FOLLOWING SHOULD CONVERT BOTH ADD AND MUL

// RUN: heir-opt --convert-elementwise-to-affine %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-ops=polynomial.add,polynomial.mul_scalar' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-ops=polynomial.add,polynomial.mul_scalar  convert-dialects=arith' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt --convert-elementwise-to-affine=convert-dialects=polynomial %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects=polynomial convert-ops=arith.addi' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_MUL %s

// THE FOLLOWING SHOULD ONLY CONVERT ADD (AND "ARITH" OPS FOR SOME, BUT THERE ARE NONE IN THE TESTS)

// RUN: heir-opt --convert-elementwise-to-affine=convert-ops=polynomial.add %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects= convert-ops=polynomial.add' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects=arith convert-ops=polynomial.add' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_ADD --check-prefix=CHECK_NOTMUL %s

// THE FOLLOWING SHOULD ONLY CONVERT MUL (AND "ARITH" OPS FOR SOME, BUT THERE ARE NONE IN THE TESTS)

// RUN: heir-opt --convert-elementwise-to-affine=convert-ops=polynomial.mul_scalar %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects= convert-ops=polynomial.mul_scalar' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_MUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects=arith convert-ops=polynomial.mul_scalar' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_MUL %s

// THE FOLLOWING SHOULD CONVERT NOTHING (EXCEPT "ARITH" OPS FOR SOME, BUT THERE ARE NONE IN THE TESTS)

// RUN: heir-opt --convert-elementwise-to-affine=convert-ops= %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt --convert-elementwise-to-affine=convert-dialects= %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-ops= convert-dialects=' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-dialects=arith' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-ops=arith.addi' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

// RUN: heir-opt '--convert-elementwise-to-affine=convert-ops=artih.addi convert-dialects=arith' %s \
// RUN: | FileCheck --enable-var-scope --check-prefix=CHECK --check-prefix=CHECK_NOTADD --check-prefix=CHECK_NOTMUL %s

!poly = !polynomial.polynomial<ring=<coefficientType = i32, coefficientModulus = 33538049 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**1024>>>

// CHECK-LABEL:  @test_elementwise
// CHECK: {{.*}} -> [[T:tensor<2x!polynomial.*33538049.*]] {
func.func @test_elementwise(%arg0: tensor<2x!poly>, %arg1: tensor<2x!poly>) ->  tensor<2x!poly> {
  // CHECK_NOTADD: polynomial.add{{.*}} : [[T]]
  %0 = polynomial.add %arg0, %arg1 : tensor<2x!poly>
  return %0 :  tensor<2x!poly>
  // CHECK_ADD: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x!polynomial.*33538049.*]]
  // CHECK_ADD: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK_ADD: [[A:%.+]] = tensor.extract %arg0[[[I]]] : [[T]]
    // CHECK_ADD: [[B:%.+]] = tensor.extract %arg1[[[I]]] : [[T]]
    // CHECK_ADD: [[S:%.+]] = polynomial.add [[A]], [[B]] : [[P:!polynomial.*33538049.*]]
    // CHECK_ADD-NOT: polynomial.add{{.*}} : [[T]]
    // CHECK_ADD: [[R:%.+]] = tensor.insert [[S]] into [[T0]][[[I]]] : [[T]]
    // CHECK_ADD: affine.yield [[R]] : [[T]]
  // CHECK_ADD: return [[LOOP]] : [[T]]
}

// CHECK-LABEL:  @test_partially_elementwise
// CHECK: ([[ARG0:%.+]]: [[T:tensor<2x!polynomial.*33538049.*]], [[ARG1:%.+]]: i32) -> [[T]] {
func.func @test_partially_elementwise(%arg0: tensor<2x!poly>, %arg1: i32) ->  tensor<2x!poly> {
  // CHECK_NOTMUL: polynomial.mul_scalar{{.*}} : [[T]], i32
  %0 = polynomial.mul_scalar %arg0, %arg1 : tensor<2x!poly>, i32
  return %0 :  tensor<2x!poly>
  // CHECK_MUL: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x!polynomial.*33538049.*]]
  // CHECK_MUL: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK_MUL: [[A:%.+]] = tensor.extract [[ARG0]][[[I]]] : [[T]]
    // CHECK_MUL: [[S:%.+]] = polynomial.mul_scalar [[A]], [[ARG1]] : [[P:!polynomial.*33538049.*]], i32
    // CHECK_MUL-NOT: polynomial.mul_scalar{{.*}} : [[T]], i32
    // CHECK_MUL: [[R:%.+]] = tensor.insert [[S]] into [[T0]][[[I]]] : [[T]]
    // CHECK_MUL: affine.yield [[R]] : [[T]]
  // CHECK_MUL: return [[LOOP]] : [[T]]
}

// CHECK-LABEL:  @test_elementwise_multidim
// CHECK: {{.*}} -> [[T:tensor<2x3x!polynomial.*33538049.*]] {
func.func @test_elementwise_multidim(%arg0: tensor<2x3x!poly>, %arg1: tensor<2x3x!poly>) ->  tensor<2x3x!poly> {
  // CHECK_NOTADD: polynomial.add{{.*}} : [[T]]
  %0 = polynomial.add %arg0, %arg1 : tensor<2x3x!poly>
  return %0 :  tensor<2x3x!poly>
  // CHECK_ADD: [[EMPTY:%.+]] = tensor.empty() : [[T:tensor<2x3x!polynomial.*33538049.*]]
  // CHECK_ADD: [[LOOP:%.+]] = affine.for [[I:%.+]] = 0 to 2 iter_args([[T0:%.+]] = [[EMPTY]]) -> ([[T]]) {
    // CHECK_ADD: [[INNERLOOP:%.+]] = affine.for [[J:%.+]] = 0 to 3 iter_args([[T1:%.+]] = [[T0]]) -> ([[T]]) {
      // CHECK_ADD: [[A:%.+]] = tensor.extract %arg0[[[I]], [[J]]] : [[T]]
      // CHECK_ADD: [[B:%.+]] = tensor.extract %arg1[[[I]], [[J]]] : [[T]]
      // CHECK_ADD: [[S:%.+]] = polynomial.add [[A]], [[B]] : [[P:!polynomial.*33538049.*]]
      // CHECK_ADD-NOT: polynomial.add{{.*}} : [[T]]
      // CHECK_ADD: [[R:%.+]] = tensor.insert [[S]] into [[T1]][[[I]], [[J]]] : [[T]]
      // CHECK_ADD: affine.yield [[R]] : [[T]]
    // CHECK_ADD: affine.yield [[INNERLOOP]] : [[T]]
  // CHECK_ADD: return [[LOOP]] : [[T]]
}
