#ifndef LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
#define LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"cheddar-configure-crypto-context"> {
  let summary = "Configure CHEDDAR crypto context from scheme parameters.";

  let description = [{
    This pass reads the `ckks.schemeParam` module attribute, converts its
    fields (logN, logDefaultScale, Q, P) into CHEDDAR-specific module
    attributes, and removes the CKKS scheme attributes.

    After this pass, the module no longer depends on the CKKS dialect.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar::CheddarDialect",
  ];

  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
  ];
}

#endif  // LIB_DIALECT_CHEDDAR_TRANSFORMS_CONFIGURECRYPTOCONTEXT_TD_
