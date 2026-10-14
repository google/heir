#ifndef LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_
#define LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_

#include <cstdint>

#include "lib/Parameters/CKKS/Params.h"
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// IWYU pragma: begin_keep
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSEnums.h"
// IWYU pragma: end_keep

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/CKKS/IR/CKKSAttributes.h.inc"

namespace mlir {
namespace heir {
namespace ckks {

// Computes ceil(log2(qi)) for each modulus in `moduli`, giving the bit width
// needed to represent each prime.
SmallVector<int64_t> getModuliBitWidths(ArrayRef<int64_t> moduli);

// Returns `attr.getModuliPerLevel()` if present, or a vector of all 1s of
// length `attr.getQ().size()` otherwise.
SmallVector<int32_t> getModuliPerLevelOrDefault(SchemeParamAttr attr);

SchemeParam getSchemeParamFromAttr(SchemeParamAttr attr);

// Parses a YAML/JSON string into a CKKS SchemeParamAttr.
FailureOr<SchemeParamAttr> parseSchemeParamAttrFromYaml(
    MLIRContext* context, llvm::StringRef yamlStr,
    llvm::function_ref<InFlightDiagnostic()> emitError);

// Loads a YAML/JSON file from `filename` into a CKKS SchemeParamAttr.
FailureOr<SchemeParamAttr> loadSchemeParamAttrFromFile(
    MLIRContext* context, llvm::StringRef filename,
    llvm::function_ref<InFlightDiagnostic()> emitError);

}  // namespace ckks
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_CKKS_IR_CKKSATTRIBUTES_H_
