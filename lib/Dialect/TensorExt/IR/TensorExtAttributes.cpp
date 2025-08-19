#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Analysis/AffineStructures.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"        // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"         // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"        // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"          // from @llvm-project

namespace mlir {
namespace heir {
namespace tensor_ext {

using presburger::IntegerPolyhedron;
using presburger::IntegerRelation;
using presburger::VarKind;

LogicalResult AlignmentAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, mlir::DenseI64ArrayAttr in,
    mlir::DenseI64ArrayAttr out, mlir::DenseI64ArrayAttr insertedDims,
    mlir::DenseI64ArrayAttr padding, TypedAttr paddingValue) {
  if (out.empty()) {
    return emitError() << "out may not be an empty array";
  }

  if (in.size() + insertedDims.size() != out.size()) {
    return emitError()
           << "in.size() + insertedDims.size() must equal out.size()";
  }

  for (auto dim : insertedDims.asArrayRef()) {
    if (dim < 0 || dim >= out.size()) {
      return emitError() << "insertedDims must be in the range [0, out.size())";
    }
  }

  for (int i = 0; i < out.size(); i++) {
    if (out[i] <= 0) {
      return emitError() << "out dimension " << i
                         << " must be positive, but was " << out[i];
    }
  }

  for (int i = 0; i < in.size(); i++) {
    if (in[i] <= 0) {
      return emitError() << "in dimension " << i
                         << " must be positive, but was " << in[i];
    }
  }

  if (!padding.empty() && padding.size() != out.size()) {
    return emitError() << "padding.size() must equal out.size()";
  }

  if (!padding.empty() && !paddingValue) {
    return emitError() << "paddingValue must be set if padding is set";
  }

  DenseSet<int64_t> insertedDimsSet(insertedDims.asArrayRef().begin(),
                                    insertedDims.asArrayRef().end());
  if (insertedDimsSet.size() != insertedDims.size()) {
    return emitError() << "insertedDims must all be unique";
  }

  // Rewrite the tensor shape with expanded dims and padding
  SmallVector<int64_t> beforeReplication;
  beforeReplication.resize(out.size(), 1);
  int inIndex = 0;
  for (int i = 0; i < out.size(); i++) {
    if (!insertedDimsSet.count(i)) {
      beforeReplication[i] = in[inIndex++];
    }
  }

  if (!padding.empty()) {
    for (int i = 0; i < out.size(); i++) {
      beforeReplication[i] += padding[i];
    }
  }

  // For each axis, input dim + padding divides or is divisible by output dim,
  // which enables replication along each axis.
  for (int i = 0; i < out.size(); i++) {
    if (beforeReplication[i] % out[i] != 0 &&
        (beforeReplication[i] == 0 || out[i] % beforeReplication[i] != 0)) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "After inserting dims and padding, each axis must have size "
            "dividing or divisible by the corresponding output axis size, but "
            "found size=";
      llvm::interleaveComma(beforeReplication, os);
      os << " and out=";
      llvm::interleaveComma(out.asArrayRef(), os);
      return emitError() << os.str();
    }
  }

  return success();
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 AffineMap map, AlignmentAttr alignment) {
  if (alignment && map.getNumDims() != alignment.getOut().size()) {
    return emitError() << "The affine map's input size (" << map.getNumDims()
                       << ") must match the "
                          "number of dimensions of alignment.out ("
                       << alignment.getOut().size() << ")";
  }

  return success();
}

void NewLayoutAttr::print(AsmPrinter& p) const {
  p << "<domainSize=" << getDomainSize();
  if (getLocalSize() > 0) {
    p << ", localSize=" << getLocalSize();
  }
  p << ", relation=\"";
  getRelation().print(p.getStream());
  p << "\">";
}

Attribute NewLayoutAttr::parse(AsmParser& parser, Type type) {
  // <domainSize=
  if (failed(parser.parseLess()) || failed(parser.parseKeyword("domainSize")) ||
      parser.parseEqual())
    return {};

  APInt parsedDomainSize(64, 1);
  if (failed(parser.parseInteger(parsedDomainSize))) {
    parser.emitError(parser.getCurrentLocation())
        << "required integer for domainSize";
    return {};
  }
  unsigned domainSize = parsedDomainSize.getZExtValue();

  // ,
  if (failed(parser.parseComma())) return {};

  // localSize=
  unsigned localSize = 0;
  if (succeeded(parser.parseOptionalKeyword("localSize"))) {
    APInt parsedLocalSize(64, 1);
    if (failed(parser.parseEqual()) ||
        failed(parser.parseInteger(parsedLocalSize))) {
      parser.emitError(parser.getCurrentLocation())
          << "required integer for localSize";
      return {};
    }
    localSize = parsedLocalSize.getZExtValue();

    if (failed(parser.parseComma())) return {};
  }

  // relation=
  if (failed(parser.parseKeyword("relation")) || parser.parseEqual()) return {};

  std::string parsedRelationString;
  if (failed(parser.parseString(&parsedRelationString))) {
    parser.emitError(parser.getCurrentLocation())
        << "expected integer relation for relation";
    return {};
  }

  IntegerSet parsedSet =
      parseIntegerSet(parsedRelationString, parser.getContext());

  if (failed(parser.parseGreater())) return {};
  return NewLayoutAttr::get(parser.getContext(), domainSize, parsedSet,
                            localSize);
}

LogicalResult NewLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, unsigned domainSize,
    IntegerSet relation, unsigned localSize) {
  // The range size (all variables except domain variables) should be 2, i.e.,
  // the ciphertext index and slot index.
  unsigned numVars = relation.getNumInputs();
  if (localSize > numVars) {
    return emitError() << "localSize (" << localSize
                       << ") must be less than or equal to the number of "
                          "variables in the relation ("
                       << numVars << ")";
  }
  if (domainSize > numVars) {
    return emitError() << "domainSize (" << domainSize
                       << ") must be less than or equal to the number of "
                          "variables in the relation ("
                       << numVars << ")";
  }
  if (domainSize + localSize > numVars) {
    return emitError() << "total number of domain and local variables ("
                       << domainSize + localSize
                       << ") must be less than or equal to the number of "
                          "variables in the relation ("
                       << numVars << ")";
  }
  if (numVars - domainSize - localSize != 2) {
    return emitError()
           << "relation must have 2 range variables, but got total vars = "
           << numVars << ", domainSize = " << domainSize
           << ", localSize = " << localSize;
  }
  return success();
}

NewLayoutAttr NewLayoutAttr::getFromIntegerRelation(
    ::mlir::MLIRContext* context, IntegerRelation relation) {
  relation.removeTrivialRedundancy();
  relation.removeDuplicateDivs();
  relation.simplify();

  std::unique_ptr<IntegerRelation> copy = relation.clone();
  copy->convertVarKind(VarKind::Domain, 0, copy->getNumDomainVars(),
                       VarKind::SetDim, 0);
  copy->convertVarKind(VarKind::Local, 0, copy->getNumLocalVars(),
                       VarKind::SetDim);
  affine::FlatAffineValueConstraints integerSet =
      IntegerPolyhedron(std::move(*copy));
  return NewLayoutAttr::get(context, relation.getNumDomainVars(),
                            integerSet.getAsIntegerSet(context),
                            relation.getNumLocalVars());
}

}  // namespace tensor_ext
}  // namespace heir
}  // namespace mlir
