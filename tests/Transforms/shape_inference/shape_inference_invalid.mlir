// RUN: heir-opt --shape-inference --split-input-file --verify-diagnostics %s 2>&1

// expected-error@+1 {{must match the type of the corresponding argument in function signature}}
func.func @mismatched_shape(%x: tensor<?xi16> {shape.shape=[42]} , %y: tensor<?xi16> {shape.shape=[44]}) -> (tensor<?xi16>) {
    // expected-error@+2 {{Operation failed to verify with its newly inferred return type(s) after its operands' types were updated during shape inference.}}
    // expected-error@+1 {{op requires the same type for all operands and results}}
    %0 = arith.addi %x, %y : tensor<?xi16>
    func.return %0 : tensor<?xi16>
}
