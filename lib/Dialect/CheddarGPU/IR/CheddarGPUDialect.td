#ifndef LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_DIALECT_TD_
#define LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_DIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def CheddarGPU_Dialect : Dialect {
  let name = "cheddar_gpu";
  let description = [{
    The `cheddar_gpu` dialect represents low-level buffer operations mirroring
    Cheddar's GPU kernels (ElementWise.cu, NTT.cu, ModSwitch.cu).

    Data representation contract:
    - Buffers are plain `memref<L x N x iW>` where L is the number of limbs,
      N = 2^log_n is the ring degree, and W in {32, 64} is the word width.
    - Limb layout is limb-major `[limb][coeff]`.
    - Limb order follows Cheddar convention: `ter[num_ter - 1 ... 0], main[0 ...], aux[0 ...]`.
    - Every buffer operand carries a required representation form:
      * `coeff`: canonical coefficient form in `[0, q)`. Natural order.
      * `coeff_centered`: centered coefficient form in `[-(q-1)/2, (q-1)/2]`. Natural order.
      * `eval`: evaluation domain representation. By Cheddar convention, data in
        `eval` form is in Montgomery form (`x * R mod q` where `R = 2^W`) AND in
        bit-reversed coefficient order.
    - Operations follow destination-passing style where output buffers are explicit
      operands and operations have no SSA results, matching Cheddar's preallocated
      device memory execution model.
  }];

  let cppNamespace = "::mlir::heir::cheddar_gpu";

  let useDefaultAttributePrinterParser = 1;
  let useStrictPropertiesInAssemblyFormat = 1;
}

#endif  // LIB_DIALECT_CHEDDARGPU_IR_CHEDDARGPU_DIALECT_TD_
