#ifndef LIB_DIALECT_KERNEL_IR_KERNELDIALECT_TD_
#define LIB_DIALECT_KERNEL_IR_KERNELDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Kernel_Dialect : Dialect {
  let name = "kernel";

  let description = [{
    The Kernel dialect contains operations for high-level math kernels.
  }];

  let cppNamespace = "::mlir::heir::kernel";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_KERNEL_IR_KERNELDIALECT_TD_
