#ifndef HEIR_LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_
#define HEIR_LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"         // from @llvm-project

namespace mlir {
namespace heir {

void registerFromAutoHogTranslation();

/// Translates the given operation from AutoHog.
OwningOpRef<Operation *> translateFromAutoHog(llvm::StringRef inputString,
                                              MLIRContext *context);

}  // namespace heir
}  // namespace mlir

#endif  // HEIR_LIB_SOURCE_AUTOHOG_AUTOHOGIMPORTER_H_
