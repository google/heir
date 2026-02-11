// RUN: heir-opt --convert-elementwise-to-affine --verify-diagnostics %s

func.func @dynamic_shape(%arg0: tensor<?xi16>, %arg1: tensor<?xi16>) -> tensor<?xi16> {
  // expected-error@+1 {{op has operand or result with dynamic shape, which is not supported by elementwise-to-affine}}
  %0 = arith.addi %arg0, %arg1 : tensor<?xi16>
  return %0 : tensor<?xi16>
}
