// RUN: heir-opt --convert-elementwise-to-linalg --one-shot-bufferize --convert-linalg-to-affine-loops

!poly = !polynomial.polynomial<<cmod=33538049, ideal=#polynomial.polynomial<1 + x**1024>>>

// CHECK-LABEL:  @test_bin_ops
// CHECK: affine.for
func.func @test_bin_ops(%arg0: tensor<2x!poly>, %arg1: tensor<2x!poly>) ->  tensor<2x!poly> {
  %0 = polynomial.add(%arg0, %arg1) : tensor<2x!poly>
  return %0 :  tensor<2x!poly>
}
