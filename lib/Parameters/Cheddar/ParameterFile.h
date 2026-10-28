#ifndef LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_
#define LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/Cheddar/IR/CheddarAttributes.h"
#include "llvm/include/llvm/ADT/StringRef.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace cheddar {

FailureOr<ParameterSetAttr> parseParameterSetFromYaml(
    MLIRContext* context, llvm::StringRef yamlStr,
    llvm::function_ref<InFlightDiagnostic()> emitError);

FailureOr<ParameterSetAttr> loadParameterSetFromFile(
    MLIRContext* context, llvm::StringRef filename,
    llvm::function_ref<InFlightDiagnostic()> emitError);

FailureOr<ckks::SchemeParamAttr> getSchemeParamAttrFromParameterSet(
    ParameterSetAttr parameterSet,
    llvm::function_ref<InFlightDiagnostic()> emitError);

LogicalResult populateSchemeParamFromParameterSet(
    ModuleOp module, ParameterSetAttr parameterSet);

}  // namespace cheddar
}  // namespace heir
}  // namespace mlir

#endif  // LIB_PARAMETERS_CHEDDAR_PARAMETERFILE_H_
