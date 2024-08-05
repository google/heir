#ifndef LIB_DIALECT_RANDOM_IR_RANDOMDIALECT_TD_
#define LIB_DIALECT_RANDOM_IR_RANDOMDIALECT_TD_

include "mlir/IR/DialectBase.td"

def Random_Dialect : Dialect {
  let name = "random";
  let description = [{
    The `random` dialect contains operations used to generate random numbers.
  }];

  let cppNamespace = "::mlir::heir::random";
  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_RANDOM_IR_RANDOMDIALECT_TD_
