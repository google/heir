#ifndef LIB_UTILS_OPASMINTERFACEHELPER_
#define LIB_UTILS_OPASMINTERFACEHELPER_

#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project

namespace mlir {
namespace heir {

void getAsmResultNames(Operation *op, ::mlir::OpAsmSetValueNameFn setNameFn);

void getAsmBlockArgumentNames(Operation *op, Region &region,
                              ::mlir::OpAsmSetValueNameFn setNameFn);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_OPASMINTERFACEHELPER_
