#ifndef LIB_TRANSFORMS_ANNOTATESCHEMEINFO_ANNOTATESCHEMEINFO_H_
#define LIB_TRANSFORMS_ANNOTATESCHEMEINFO_ANNOTATESCHEMEINFO_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/AnnotateSchemeInfo/AnnotateSchemeInfo.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_ANNOTATESCHEMEINFO_ANNOTATESCHEMEINFO_H_
