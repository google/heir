#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_TD_

include "mlir/Pass/PassBase.td"

def LWEToStableHLO : Pass<"lwe-to-stablehlo"> {
  let summary = "Lower `lwe` operations to `stablehlo` canonical rank-5 tensors.";

  let description = [{
    Lowers RLWE (`lwe`) ciphertext operations directly to `stablehlo` tensor
    operations using a canonical rank-5 `ui32` tensor layout
    `tensor<batch x size x r x c x num_moduli x ui32>`, where `r * c == N` is
    the 2D-folded polynomial ring degree and `num_moduli` is the number of
    active `< 2^32` RNS prime limbs.

    Scalar `!lwe.lwe_ciphertext` values use the `batch` option (default `1`),
    while 1D `tensor<B x !lwe.lwe_ciphertext>` values use `B` as the batch
    dimension. When `r` and `c` are both `0` (default), they are auto-computed
    as `r = 2^(floor(log2(N) / 2))` and `c = N / r`; if only one of `r` or `c`
    is `0`, it is auto-computed as `N` divided by the other dimension.
  }];

  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect",
  ];

  let options = [
    Option<"r", "r", "int",
           /*default=*/"0",
           "TPU tiling row dimension (0 = auto-compute from ring degree N)">,
    Option<"c", "c", "int",
           /*default=*/"0",
           "TPU tiling column dimension (0 = auto-compute from ring degree N)">,
    Option<"batch", "batch", "int",
           /*default=*/"1",
           "Default batch dimension for scalar ciphertext values">,
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOSTABLEHLO_LWETOSTABLEHLO_TD_
