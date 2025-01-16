#ifndef LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_
#define LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_TD_

include "mlir/Pass/PassBase.td"

def ArithToCGGIQuart : Pass<"arith-to-cggi-quart"> {
  let summary = "Lower `arith` to `cggi` dialect and divide each operation into smaller parts.";
    let description = [{
    This pass converts high precision arithmetic operations, i.e. operations on 32 bit integer,
    into a sequence of lower precision operations, i.e 8b operations.
    Currently, the pass splits the 32b integer into four 8b integers, using the tensor dialect.
    These smaller integers are stored in an 16b integer, so that we don't lose the carry information.
    This pass converts the `arith` dialect to the `cggi` dialect.

    Based on the `arith-emulate-wide-int` pass from the MLIR arith dialect.

    General assumption: the first element in the tensor is also the LSB element.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::cggi::CGGIDialect",
  ];
}

#endif  // LIB_DIALECT_ARITH_CONVERSIONS_ARITHTOCGGIQUART_ARITHTOCGGIQUART_TD_
