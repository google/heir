#ifndef LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
#define LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_

include "mlir/Pass/PassBase.td"

def AnnotateModule : Pass<"annotate-module"> {
  let summary = "Annotate ModuleOp with Scheme And/Or Backend";
  let description = [{
    This pass annotates the module with a scheme and/or backend.

    This pass should be called before all lowering to enable lowering
    to the desired scheme and backend.

    Available scheme:
      - `bgv`
      - `ckks`
      - `bfv`
      - `cggi`

    Available backend:
      - `openfhe`
      - `lattigo`

    (* example filepath=tests/Transforms/annotate_module/pass.mlir *)
  }];

  let options = [
    Option<"scheme", "scheme", "std::string", /*default=*/"\"\"",
           "The scheme to annotate the module with.">,
    Option<"backend", "backend", "std::string", /*default=*/"\"\"",
            "The backend to annotate the module with.">,
  ];
}

#endif  // LIB_TRANSFORMS_ANNOTATEMODULE_ANNOTATEMODULE_TD_
