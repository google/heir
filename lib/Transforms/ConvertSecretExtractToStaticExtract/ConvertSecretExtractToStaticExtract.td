#ifndef LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_
#define LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretExtractToStaticExtract : Pass<"convert-secret-extract-to-static-extract"> {
  let summary = "Convert `tensor.extract` operations on secret index to static extract operations.";
  let description = [{
    Converts `tensor.extract` operations that read value at secret index to
    alternative static `tensor.extract` operations that extracts value at each
    index and conditionally selects the value extracted at the secret index.

    Note: Running this pass alone does not result in a data-oblivious program;
    we have to run the `--convert-if-to-select` pass to the resulting program
    to convert the secret-dependent If-operation to a Select-operation.

    (* example filepath=tests/Transforms/convert_secret_extract_to_static_extract/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETEXTRACTTOSTATICEXTRACT_CONVERTSECRETEXTRACTTOSTATICEXTRACT_TD_
