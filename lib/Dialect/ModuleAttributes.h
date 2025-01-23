#ifndef LIB_DIALECT_MODULEATTRIBUTES_H_
#define LIB_DIALECT_MODULEATTRIBUTES_H_

#include <string>

#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project

constexpr const static ::llvm::StringLiteral kBGVSchemeAttrName = "scheme.bgv";
constexpr const static ::llvm::StringLiteral kCKKSSchemeAttrName =
    "scheme.ckks";
constexpr const static ::llvm::StringLiteral kCGGISchemeAttrName =
    "scheme.cggi";

#endif  // LIB_DIALECT_MODULEATTRIBUTES_H_
