#ifndef LIB_DIALECT_CHEDDAR_IR_CHEDDARDIALECT_TD_
#define LIB_DIALECT_CHEDDAR_IR_CHEDDARDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Cheddar_Dialect : Dialect {
  let name = "cheddar";
  let description = [{
    The `cheddar` dialect is an exit dialect for generating C++ code against the
    CHEDDAR GPU FHE library API.

    CHEDDAR is a CKKS-only GPU-accelerated FHE library. It supports both 32-bit
    and 64-bit word types, with 32-bit being the primary fast path on GPUs.

    See https://github.com/scale-snu/cheddar-fhe
  }];

  let cppNamespace = "::mlir::heir::cheddar";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif  // LIB_DIALECT_CHEDDAR_IR_CHEDDARDIALECT_TD_
