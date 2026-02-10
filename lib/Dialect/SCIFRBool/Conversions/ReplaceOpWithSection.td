#ifndef LIB_DIALECT_SCIFRBOOL_CONVERSIONS_REPLACEOPWITHSECTION_TD_
#define LIB_DIALECT_SCIFRBOOL_CONVERSIONS_REPLACEOPWITHSECTION_TD_

include "mlir/Pass/PassBase.td"

def ReplaceOpWithSection : Pass<"replace-op-with-section"> {
  let summary = "Move each CGGI/SCIFRBool Operation inside a Section Operation";
  let description = [{
      Move each CGGI/SCIFRBool Operation inside a Section Operation.
      We generate mlir with sections for larger boolean circuits.
  }];

  let dependentDialects = [
    "mlir::heir::cggi::CGGIDialect",
    "mlir::scifrbool::SCIFRBoolDialect"
  ];
  let options = [
    Option<"nChips", "nChips", "uint32_t", "2", "number of available chips">
  ];
}

#endif  // LIB_DIALECT_SCIFRBOOL_CONVERSIONS_REPLACEOPWITHSECTION_TD_
