#include "lib/Dialect/FuncUtils.h"

#include <cassert>

#include "llvm/include/llvm/ADT/StringExtras.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
namespace heir {

/// Copied from MLIR AsmPrinter.cpp with modification.
/// Sanitize the given name such that it can be used as a valid identifier. If
/// the string needs to be modified in any way, the provided buffer is used to
/// store the new copy,
StringRef sanitizeIdentifier(StringRef name, SmallString<16>& buffer,
                             StringRef allowedPunctChars) {
  assert(!name.empty() && "Shouldn't have an empty name here");

  auto validChar = [&](char ch) {
    return llvm::isAlnum(ch) || allowedPunctChars.contains(ch);
  };

  auto copyNameToBuffer = [&] {
    for (char ch : name) {
      if (validChar(ch))
        buffer.push_back(ch);
      else if (ch == ' ' || ch == '<' || ch == '>')
        buffer.push_back('_');
      else
        buffer.append(llvm::utohexstr((unsigned char)ch));
    }
  };

  // Check to see that the name consists of only valid identifier characters.
  for (char ch : name) {
    if (!validChar(ch)) {
      copyNameToBuffer();
      return buffer;
    }
  }

  // If there are no invalid characters, return the original name.
  return name;
}

}  // namespace heir
}  // namespace mlir
