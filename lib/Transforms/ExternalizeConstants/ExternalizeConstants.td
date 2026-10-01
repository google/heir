#ifndef LIB_TRANSFORMS_EXTERNALIZECONSTANTS_EXTERNALIZECONSTANTS_TD_
#define LIB_TRANSFORMS_EXTERNALIZECONSTANTS_EXTERNALIZECONSTANTS_TD_

include "mlir/Pass/PassBase.td"

def ExternalizeConstants : Pass<"externalize-constants"> {
  let summary = "Externalize large constants to binary files";
  let description = [{
    Replaces arith.constant operations containing large dense tensors with
    `preprocessing.load_resource` ops, and writes the constant data to external
    binary files.
  }];

  let options = [
    Option<"thresholdElements", "threshold-elements", "int64_t",
           /*default=*/"1024", "Minimum number of elements to externalize">,
    Option<"outputDir", "output-dir", "std::string",
           /*default=*/"\".\"", "Directory where binary files are written">,
    Option<"runtimeLoadDir", "runtime-load-dir", "std::string",
           /*default=*/"\".\"", "Directory path used in the generated load_resource op">
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::preprocessing::PreprocessingDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // LIB_TRANSFORMS_EXTERNALIZECONSTANTS_EXTERNALIZECONSTANTS_TD_
