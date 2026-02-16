#ifndef LIB_TABLEGEN_COMPILATIONTARGETEMITTER_H_
#define LIB_TABLEGEN_COMPILATIONTARGETEMITTER_H_

#include "llvm/include/llvm/TableGen/Record.h"  // from @llvm-project

namespace mlir {
namespace heir {

bool emitCompilationTargetRegistration(const llvm::RecordKeeper& records,
                                       llvm::raw_ostream& os);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TABLEGEN_COMPILATIONTARGETEMITTER_H_
