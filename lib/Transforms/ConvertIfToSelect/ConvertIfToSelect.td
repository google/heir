#ifndef LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_
#define LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_

include "mlir/Pass/PassBase.td"

def ConvertIfToSelect : Pass<"convert-if-to-select"> {
  let summary = "Convert scf.if operations on secret conditions to arith.select operations.";
  let description = [{ Conversion for If-operations that evaluate secret condition to alternative select operations. }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTIFTOSELECT_CONVERTIFTOSELECT_TD_
