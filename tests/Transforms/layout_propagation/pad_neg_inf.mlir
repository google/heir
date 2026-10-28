// RUN: heir-opt --layout-propagation=min-slot-count=16 --verify-diagnostics %s

func.func @pad_neg_inf(%arg0: !secret.secret<tensor<5xf32>>) -> !secret.secret<tensor<8xf32>> {
  %0 = secret.generic(%arg0: !secret.secret<tensor<5xf32>>) {
  ^body(%input0: tensor<5xf32>):
    %neg_inf = arith.constant 0xFF800000 : f32
    // expected-error@+2 {{layout propagation only supports zero-padding tensor.pad}}
    // expected-note@+1 {{replaces the -inf padding of a max pool with zero only when the pooled input is nonnegative}}
    %padded = tensor.pad %input0 low[2] high[1] {
    ^body(%arg1: index):
      tensor.yield %neg_inf : f32
    } : tensor<5xf32> to tensor<8xf32>
    secret.yield %padded : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
}
