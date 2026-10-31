#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_PASSES_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

def LWEToCheddarGPU : Pass<"lwe-to-cheddar-gpu", "mlir::ModuleOp"> {
  let summary = "Lower `lwe` and `ckks` to `cheddar_gpu` destination-passing memref ops.";

  let description = [{
    This pass lowers CKKS scheme-level operations to Cheddar GPU dialect operations
    in destination-passing style on device-resident memrefs.
  }];

  let dependentDialects = [
    "mlir::heir::cheddar_gpu::CheddarGPUDialect",
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
  ];

  let options = [
    Option<"paramsPath", "params", "std::string", /*default=*/"",
           "Path to Cheddar parameter JSON or YAML file.">,
    Option<"fuseRelinRescale", "fuse-relin-rescale", "bool", /*default=*/"true",
           "Fuse adjacent relinearize and rescale into relinearize_rescale.">,
    Option<"splatAsConst", "splat-as-const", "bool", /*default=*/"true",
           "Lower splat constant plaintext operations to *_const operations.">,
    Option<"encodeMode", "encode", "std::string", /*default=*/"\"args\"",
           "Encoding mode for plaintexts: 'args' (hoisted arguments) or 'runtime'.">,
    Option<"emitOracleTrace", "emit-oracle-trace", "bool", /*default=*/"true",
           "Emit JSON-lines trace of Cheddar Context operations as a module attribute/global.">,
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOCHEDDARGPU_PASSES_TD_
