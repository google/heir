#ifndef LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_
#define LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_

include "mlir/Pass/PassBase.td"

def ConvertSecretInsertToStaticInsert : Pass<"convert-secret-insert-to-static-insert"> {
  let summary = "Convert `tensor.insert` operations on secret index to static insert operations.";
  let description = [{
    Converts `tensor.insert` operations that write to secret index to
    alternative static `tensor.insert` operations that inserts the inserted
    value at each index and conditionally selects the newly produced tensor
    that contains the value at the secret index.

    Note: Running this pass alone does not result in a data-oblivious program;
    we have to run the `--convert-if-to-select` pass to the resulting program
    to convert the secret-dependent If-operation to a Select-operation.

    (* example filepath=tests/Transforms/convert_secret_insert_to_static_insert/doctest.mlir *)
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_CONVERTSECRETINSERTTOSTATICINSERT_CONVERTSECRETINSERTTOSTATICINSERT_TD_
