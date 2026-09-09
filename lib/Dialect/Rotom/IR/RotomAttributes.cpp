#include "lib/Dialect/Rotom/IR/RotomAttributes.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/Sequence.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace rotom {

size_t inferCtPrefixLen(ArrayRef<DimAttr> dims, int64_t n) {
  int64_t nRem = n;
  size_t i = dims.size();
  while (i > 0) {
    if (nRem <= 1) break;
    DimAttr d = dims[i - 1];
    const int64_t sz = d.getSize();
    if (sz <= 0) break;
    if (sz <= nRem && (nRem % sz == 0)) {
      nRem /= sz;
      --i;
      continue;
    }
    break;
  }
  while (i > 0 && dims[i - 1].getSize() == 1) --i;
  return i;
}

// Preprocesses a layout (`dims`, slot count `n`) into the `LayoutData`
// descriptor used to emit ciphertext addresses; also the validity check
// behind `LayoutAttr::verify`. `pieces` holds the traversal, replication,
// and gap dimensions in packing order; see `LayoutData` in the header for
// exact field semantics.
static FailureOr<LayoutData> preprocessLayoutData(ArrayAttr dims, int64_t n,
                                                  MLIRContext* ctx) {
  LayoutData data;
  data.n = n;
  if (data.n <= 0) return failure();

  std::map<int64_t, DimAttr> axisForDim;
  SmallVector<DimAttr> writtenDims;
  writtenDims.reserve(dims.size());
  data.pieces.reserve(dims.size());
  for (Attribute a : dims) {
    auto d = dyn_cast<DimAttr>(a);
    if (!d) return failure();
    writtenDims.push_back(d);
    if (d.isGap()) {
      data.pieces.push_back({d, LayoutPieceKind::Gap});
      continue;
    }
    if (d.isReplicate()) {
      data.pieces.push_back({d, LayoutPieceKind::Replication});
      continue;
    }
    // A contraction spans no axis of the laid-out value, so it measures like
    // a gap; only the code reading a `compute` placement tells the two apart.
    if (d.isContraction()) {
      data.pieces.push_back({d, LayoutPieceKind::Gap});
      continue;
    }
    if (d.getDim() < 0) return failure();
    axisForDim.try_emplace(d.getDim(), d);
    // axisIndex is set below: a dim's rank isn't known until all dims are seen.
    data.pieces.push_back({d, LayoutPieceKind::Traversal, /*axisIndex=*/-1,
                           /*divBy=*/d.getStride()});
  }

  // Number the axes by ascending dim id (the order std::map gives).
  llvm::DenseMap<int64_t, int64_t> axisIndexForDim;
  for (auto& [dim, dimAttr] : axisForDim) {
    axisIndexForDim[dim] = static_cast<int64_t>(data.axes.size());
    data.axes.push_back(dimAttr);
  }
  for (LayoutPiece& piece : data.pieces) {
    if (piece.kind == LayoutPieceKind::Traversal) {
      piece.axisIndex = axisIndexForDim[piece.dim.getDim()];
    }
  }

  // Count pieces per tensor dim and the dim's full extent.
  llvm::DenseMap<int64_t, int64_t> pieceCount;
  llvm::DenseMap<int64_t, int64_t> dimFullExtent;
  for (const LayoutPiece& piece : data.pieces) {
    if (piece.kind != LayoutPieceKind::Traversal) continue;
    ++pieceCount[piece.dim.getDim()];
    auto [it, inserted] = dimFullExtent.try_emplace(piece.dim.getDim(), 1);
    it->second *= piece.dim.getSize();
  }

  for (LayoutPiece& piece : data.pieces) {
    if (piece.kind != LayoutPieceKind::Traversal) continue;
    if (pieceCount[piece.dim.getDim()] == 1) {
      // Lone piece: (i / 1) mod 0 = i.
      piece.divBy = 1;
      piece.modBy = 0;
    } else {
      // Split piece: it reads (i / stride) mod extent of the axis index.
      const int64_t extent = piece.dim.getSize();
      const int64_t full = dimFullExtent[piece.dim.getDim()];
      piece.modBy = (piece.divBy * extent < full) ? extent : 0;
    }
  }

  // Each multi-piece tensor dim must be a valid split:
  // sorted by stride, the divisors are the cumulative products of the lower
  // extents (1, e0, e0*e1, ...), and the extents multiply to the full extent.
  for (auto& [dim, count] : pieceCount) {
    if (count == 1) continue;
    SmallVector<std::pair<int64_t, int64_t>> parts;  // (stride, extent)
    for (const LayoutPiece& piece : data.pieces) {
      if (piece.kind == LayoutPieceKind::Traversal &&
          piece.dim.getDim() == dim) {
        parts.push_back({piece.divBy, piece.dim.getSize()});
      }
    }
    llvm::sort(parts);  // ascending stride
    int64_t expected = 1;
    for (auto [stride, extent] : parts) {
      if (extent <= 0 || stride != expected) return failure();
      expected *= extent;
    }
    if (expected != dimFullExtent[dim]) return failure();
  }

  for (size_t ti = 0; ti < data.axes.size(); ++ti) {
    const int64_t dim = data.axes[ti].getDim();
    data.axes[ti] = DimAttr::get(ctx, dim, dimFullExtent[dim], /*stride=*/1);
  }

  data.ctPrefixLen =
      static_cast<int64_t>(inferCtPrefixLen(writtenDims, data.n));

  return data;
}

// The dim position is written `R` for replication and `G` for gap (the
// readable forms, also how the printer emits them); the numeric ids -1 and
// -2 are still accepted.
static ParseResult parseDimTripleAfterLSquare(AsmParser& parser, int64_t& dim,
                                              int64_t& size, int64_t& stride) {
  if (succeeded(parser.parseOptionalKeyword("R"))) {
    dim = -1;
  } else if (succeeded(parser.parseOptionalKeyword("G"))) {
    dim = -2;
  } else if (succeeded(parser.parseOptionalKeyword("K"))) {
    dim = kDimContraction;
  } else if (parser.parseInteger(dim)) {
    return failure();
  }
  if (parser.parseColon() || parser.parseInteger(size)) return failure();
  stride = 1;
  if (succeeded(parser.parseOptionalColon()) && parser.parseInteger(stride)) {
    return failure();
  }
  return parser.parseRSquare();
}

static ParseResult parseDimTriple(AsmParser& parser, int64_t& dim,
                                  int64_t& size, int64_t& stride) {
  return failure(parser.parseLSquare() ||
                 failed(parseDimTripleAfterLSquare(parser, dim, size, stride)));
}

static void printDimTriple(AsmPrinter& printer, DimAttr dim) {
  printer << "[";
  if (dim.isReplicate()) {
    printer << "R";
  } else if (dim.isContraction()) {
    printer << "K";
  } else if (dim.isGap()) {
    printer << "G";
  } else {
    printer << dim.getDim();
  }
  printer << ":" << dim.getSize() << ":" << dim.getStride() << "]";
}

// Parses `[piece, ... | piece, ...]`: the `|` separates the ciphertext dims
// from the slot dims (absent when there are no ciphertext dims). The written
// boundary is returned in `writtenCtLen` for validation against the derived
// split.
static ParseResult parseLayoutDims(AsmParser& parser,
                                   SmallVector<Attribute>& dims,
                                   std::optional<int64_t>& writtenCtLen) {
  if (parser.parseLSquare()) return failure();
  if (succeeded(parser.parseOptionalRSquare())) return success();

  while (true) {
    if (succeeded(parser.parseOptionalLSquare())) {
      int64_t dim;
      int64_t size;
      int64_t stride;
      if (failed(parseDimTripleAfterLSquare(parser, dim, size, stride)))
        return failure();
      dims.push_back(DimAttr::get(parser.getContext(), dim, size, stride));
    } else {
      Attribute dim;
      if (parser.parseAttribute(dim)) return failure();
      if (!isa<DimAttr>(dim)) {
        return parser.emitError(parser.getNameLoc())
               << "expected a #rotom.dim attribute";
      }
      dims.push_back(dim);
    }

    if (succeeded(parser.parseOptionalComma())) continue;
    if (succeeded(parser.parseOptionalVerticalBar())) {
      if (writtenCtLen.has_value()) {
        return parser.emitError(parser.getNameLoc())
               << "at most one `|` may separate ciphertext dims from slot "
                  "dims";
      }
      writtenCtLen = static_cast<int64_t>(dims.size());
      continue;
    }
    return parser.parseRSquare();
  }
}

// One argument of a roll pair: a bare non-negative integer is a dims-list
// position (one piece); `axis N` names the whole tensor axis N.
static ParseResult parseRollArg(AsmParser& parser, int64_t& encoded) {
  const bool isAxis = succeeded(parser.parseOptionalKeyword("axis"));
  int64_t value;
  if (parser.parseInteger(value)) return failure();
  if (value < 0) {
    return parser.emitError(parser.getNameLoc())
           << (isAxis ? "an axis roll argument must name a non-negative "
                        "tensor axis"
                      : "a piece roll argument must be a non-negative dims "
                        "position (write a whole-axis argument as `axis N`)");
  }
  encoded = encodeRollArg({isAxis, value});
  return success();
}

static ParseResult parseLayoutRolls(AsmParser& parser,
                                    SmallVector<int64_t>& rolls) {
  if (parser.parseLSquare()) return failure();
  if (succeeded(parser.parseOptionalRSquare())) return success();

  // Each entry is one `(from, by)` pair, parenthesized -- the form the
  // printer emits, so written and round-tripped layouts read alike.
  while (true) {
    int64_t from;
    int64_t by;
    if (parser.parseLParen() || failed(parseRollArg(parser, from)) ||
        parser.parseComma() || failed(parseRollArg(parser, by)) ||
        parser.parseRParen()) {
      return failure();
    }
    rolls.push_back(from);
    rolls.push_back(by);

    if (succeeded(parser.parseOptionalComma())) continue;
    return parser.parseRSquare();
  }
}

static LogicalResult verifyLayoutRolls(
    ArrayAttr dims, DenseI64ArrayAttr rolls,
    function_ref<InFlightDiagnostic()> emitError) {
  const bool noRolls = !rolls || rolls.empty();
  if (noRolls) return success();
  ArrayRef<int64_t> r = rolls.asArrayRef();
  if (r.size() % 2 != 0) {
    return emitError() << "rolls must contain an even number of arguments "
                          "(pairs)";
  }

  for (size_t i = 0; i < r.size(); i += 2) {
    const RollArg from = decodeRollArg(r[i]);
    const RollArg by = decodeRollArg(r[i + 1]);

    // Resolve each argument: the piece it names (null for axis arguments)
    // and the tensor axis it reads or rewrites (sentinel for gap/replication
    // pieces).
    DimAttr fromPiece;
    DimAttr byPiece;
    int64_t fromAxis = 0;
    int64_t byAxis = 0;
    auto checkArg = [&](const RollArg& e, DimAttr& piece,
                        int64_t& axisId) -> LogicalResult {
      if (e.isAxis) {
        // The piece count decides how the axis may be named: an `axis`
        // argument is legal only for a split axis.
        const AxisPieces pieces = axisPieces(dims, e.index);
        if (pieces.count == 0) {
          return emitError() << "an axis roll argument must name a tensor "
                                "axis present in dims";
        }
        if (!pieces.isSplit()) {
          return emitError() << "an axis roll argument requires a split "
                                "axis; write an unsplit axis's argument as "
                                "its piece position";
        }
        axisId = e.index;
        return success();
      }
      if (e.index >= static_cast<int64_t>(dims.size())) {
        return emitError() << "roll piece argument out of bounds for dims "
                              "list";
      }
      piece = dyn_cast<DimAttr>(dims[e.index]);
      if (!piece) {
        return emitError() << "roll arguments must refer to #rotom.dim "
                              "entries";
      }
      axisId = piece.getDim();
      return success();
    };
    if (failed(checkArg(from, fromPiece, fromAxis)) ||
        failed(checkArg(by, byPiece, byAxis))) {
      return failure();
    }

    // The extents need not match: a roll rewrites the from index to
    // (idx - shift) mod extent(from), well-defined for any partner extent (a
    // smaller partner covers a prefix of the rotations, a larger one wraps).
    // FROM is the index expression being rewritten, so it must be a
    // traversal piece or a whole (traversal) axis. The by argument may be
    // any kind: rolling by a replication or gap piece shifts by that piece's
    // block index, so each block holds a distinct cyclic rotation of the
    // rolled index -- the layout materializes every rotation and alignment
    // becomes block selection. (A rolled-by gap thus claims its blocks,
    // unlike a plain gap.)
    if (!from.isAxis && (fromPiece.isGap() || fromPiece.isReplicate())) {
      return emitError() << "the rolled dim must be a traversal dim (dim >= "
                            "0)";
    }
    // A roll may not shift an index by an argument on its own axis: an axis
    // FROM rewrites every piece of the axis, including the one the by
    // argument reads,
    // and a piece FROM taking a by on the same axis is a self-roll no
    // packing needs. Gap and replication by arguments name no axis, so they
    // never collide.
    if (fromAxis == byAxis) {
      return emitError()
             << "a roll may not shift an index by an argument on the same "
                "axis";
    }
    // A rolled-by GAP claims one ciphertext block per gap index, each holding
    // a distinct rotation of the rolled index. If the gap is larger than the
    // rolled extent the rotations repeat (period = the from extent), claiming
    // blocks the conversion/kernel accounting was never audited for.
    // (Replication partners of larger extent are intended -- replicate-then-
    // roll -- so only gaps are bounded.)
    const int64_t fromExtent =
        from.isAxis ? axisPieces(dims, fromAxis).extent : fromPiece.getSize();
    if (byPiece && byPiece.isGap() && byPiece.getSize() > fromExtent) {
      return emitError() << "a rolled-by gap dim must not exceed the rolled "
                            "dim's extent";
    }
  }
  return success();
}

void DimAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printDimTriple(printer, *this);
  printer << ">";
}

Attribute DimAttr::parse(AsmParser& parser, Type type) {
  int64_t dim;
  int64_t size;
  int64_t stride;

  if (parser.parseLess() || failed(parseDimTriple(parser, dim, size, stride)) ||
      parser.parseGreater()) {
    return {};
  }

  return DimAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), dim, size, stride);
}

LogicalResult DimAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t dim, int64_t size, int64_t stride) {
  if (dim < kDimContraction) {
    return emitError() << "`dim` must be >= -3, got " << dim;
  }
  if (size <= 0) {
    return emitError() << "`size` must be > 0, got " << size;
  }
  if (stride <= 0) {
    return emitError() << "`stride` must be > 0, got " << stride;
  }
  return success();
}

void LayoutAttr::print(AsmPrinter& printer) const {
  printer << "<n = " << getN();

  DenseI64ArrayAttr rolls = getRolls();
  if (rolls && !rolls.asArrayRef().empty()) {
    ArrayRef<int64_t> values = rolls.asArrayRef();
    auto printArg = [&](int64_t encoded) {
      const RollArg e = decodeRollArg(encoded);
      if (e.isAxis) printer << "axis ";
      printer << e.index;
    };
    printer << ", rolls = [";
    for (size_t i = 0; i < values.size(); i += 2) {
      if (i != 0) printer << ", ";
      printer << "(";
      printArg(values[i]);
      printer << ", ";
      printArg(values[i + 1]);
      printer << ")";
    }
    printer << "]";
  }

  SmallVector<DimAttr> dimVec;
  dimVec.reserve(getDims().size());
  for (Attribute attr : getDims()) dimVec.push_back(cast<DimAttr>(attr));
  const size_t ctLen = inferCtPrefixLen(dimVec, getN());

  printer << ", dims = [";
  for (size_t i = 0; i < dimVec.size(); ++i) {
    if (i > 0) printer << (i == ctLen ? " | " : ", ");
    printDimTriple(printer, dimVec[i]);
  }
  printer << "]>";
}

Attribute LayoutAttr::parse(AsmParser& parser, Type type) {
  int64_t n;
  SmallVector<int64_t> rolls;
  SmallVector<Attribute> dims;
  std::optional<int64_t> writtenCtLen;

  if (parser.parseLess()) return {};

  if (succeeded(parser.parseOptionalKeyword("n"))) {
    if (parser.parseEqual() || parser.parseInteger(n) || parser.parseComma()) {
      return {};
    }

    if (succeeded(parser.parseOptionalKeyword("rolls"))) {
      if (parser.parseEqual() || failed(parseLayoutRolls(parser, rolls)) ||
          parser.parseComma()) {
        return {};
      }
    }

    if (parser.parseKeyword("dims") || parser.parseEqual() ||
        failed(parseLayoutDims(parser, dims, writtenCtLen)) ||
        parser.parseGreater()) {
      return {};
    }
  } else if (succeeded(parser.parseOptionalKeyword("dims"))) {
    if (parser.parseEqual() ||
        failed(parseLayoutDims(parser, dims, writtenCtLen)) ||
        parser.parseComma() || parser.parseKeyword("n") ||
        parser.parseEqual() || parser.parseInteger(n)) {
      return {};
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseKeyword("rolls") || parser.parseEqual() ||
          failed(parseLayoutRolls(parser, rolls))) {
        return {};
      }
    }

    if (parser.parseGreater()) return {};
  } else {
    parser.emitError(parser.getNameLoc())
        << "expected `n` or `dims` in rotom layout";
    return {};
  }

  // Verify the written `|` boundary.
  SmallVector<DimAttr> dimVec;
  dimVec.reserve(dims.size());
  for (Attribute attr : dims) dimVec.push_back(cast<DimAttr>(attr));
  const int64_t derivedCtLen =
      static_cast<int64_t>(inferCtPrefixLen(dimVec, n));
  if (n > 0 && writtenCtLen.value_or(0) != derivedCtLen) {
    parser.emitError(parser.getNameLoc())
        << "the written `|` ciphertext/slot split (" << writtenCtLen.value_or(0)
        << " ciphertext dims) does not match the derived split ("
        << derivedCtLen
        << "): the slot side is the longest dims suffix whose extents fit "
           "n = "
        << n;
    return {};
  }

  MLIRContext* context = parser.getContext();
  return LayoutAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, context,
      ArrayAttr::get(context, dims), n, DenseI64ArrayAttr::get(context, rolls));
}

namespace {
template <typename RangeT>
AxisPieces accumulateAxisPieces(RangeT&& dims, int64_t axis) {
  AxisPieces pieces;
  for (DimAttr d : dims) {
    if (d.isGap() || d.isReplicate() || d.getDim() != axis) continue;
    ++pieces.count;
    pieces.extent *= d.getSize();
  }
  return pieces;
}
}  // namespace

AxisPieces axisPieces(ArrayRef<DimAttr> dims, int64_t axis) {
  return accumulateAxisPieces(dims, axis);
}

AxisPieces axisPieces(ArrayAttr dims, int64_t axis) {
  return accumulateAxisPieces(dims.getAsRange<DimAttr>(), axis);
}

SmallVector<RollSpec> getRollSpecs(LayoutAttr layout) {
  SmallVector<RollSpec> specs;
  DenseI64ArrayAttr rolls = layout.getRolls();
  if (!rolls) return specs;
  ArrayRef<int64_t> r = rolls.asArrayRef();

  for (size_t i = 0; i + 1 < r.size(); i += 2) {
    specs.push_back({decodeRollArg(r[i]), decodeRollArg(r[i + 1])});
  }
  return specs;
}

FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr layout) {
  return preprocessLayoutData(layout.getDims(), layout.getN(),
                              layout.getContext());
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayAttr dims, int64_t n,
                                 DenseI64ArrayAttr rolls) {
  if (n <= 0) {
    return emitError() << "`n` must be > 0, got " << n;
  }
  auto preprocessed = preprocessLayoutData(dims, n, dims.getContext());
  if (failed(preprocessed)) {
    return emitError() << "`dims` must be an array of `#rotom.dim<...>`";
  }

  if (failed(verifyLayoutRolls(dims, rolls, emitError))) {
    return failure();
  }

  SmallVector<DimAttr> dimVec;
  dimVec.reserve(dims.size());
  for (Attribute attr : dims) dimVec.push_back(cast<DimAttr>(attr));
  const size_t ctLen = inferCtPrefixLen(dimVec, n);
  int64_t slotExtent = 1;
  for (size_t p = ctLen; p < dimVec.size(); ++p) {
    DimAttr d = dimVec[p];
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getSize()))) {
      return emitError() << "slot dim size must be a power of two, got "
                         << d.getSize();
    }
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getStride()))) {
      return emitError() << "slot dim stride must be a power of two, got "
                         << d.getStride();
    }
    slotExtent *= d.getSize();
  }

  // The slot side must fill the ciphertext exactly. Unused slots must be
  // represented as an explicit gap piece.
  if (slotExtent != n) {
    return emitError() << "slot dims must fill the ciphertext exactly (slot "
                          "extent "
                       << slotExtent << " vs n = " << n
                       << "); state unused capacity as an explicit gap piece";
  }

  return success();
}

LogicalResult SeedAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                               ArrayAttr layouts) {
  for (Attribute layout : layouts) {
    auto layoutAttr = dyn_cast<LayoutAttr>(layout);
    if (!layoutAttr) {
      return emitError() << "seed layouts must be `rotom.layout` attributes";
    }
    if (failed(LayoutAttr::verify(emitError, layoutAttr.getDims(),
                                  layoutAttr.getN(), layoutAttr.getRolls())))
      return failure();
  }
  return success();
}

void canonicalizeLayoutDims(MLIRContext* ctx, SmallVector<DimAttr>& dims,
                            int64_t n, SmallVector<int64_t>& rolls) {
  // A unit axis -- one whose whole extent is one -- places no data: its index
  // has a single value, so it contributes nothing to any address, and a
  // factor of one changes no other piece's offset. Its position was therefore
  // free, which is what let one packing have several forms and made every
  // consumer walk past unit pieces. Pin it to the tail in axis order, where
  // addUnitAxisPieces already puts the pieces it creates and where
  // inferCtPrefixLen's trailing trim already assumes they live. The piece
  // stays: the relation the layout lowers to needs one domain variable per
  // tensor axis. This rewrites the form, never the packing.
  {
    llvm::DenseMap<int64_t, int64_t> extentOfAxis;
    for (DimAttr d : dims) {
      if (d.isGap() || d.isReplicate()) continue;
      auto it = extentOfAxis.find(d.getDim());
      extentOfAxis[d.getDim()] =
          it == extentOfAxis.end() ? d.getSize() : it->second * d.getSize();
    }
    auto isUnitAxisPiece = [&](DimAttr d) {
      if (d.isGap() || d.isReplicate()) return false;
      auto it = extentOfAxis.find(d.getDim());
      return it != extentOfAxis.end() && it->second == 1;
    };
    // Extent-one replication and gaps are NOT dropped: alignment pairs two
    // layouts by piece count (pieceMovements requires the counts to match),
    // so a piece like [R:1:1] is a positional placeholder even though it
    // places no data. Dropping it loses the matmul kernel entirely.
    auto isInert = [](DimAttr d) {
      (void)d;
      return false;
    };
    SmallVector<DimAttr> rest, units;
    SmallVector<int64_t> restOld, unitOld;
    bool dropped = false;
    for (auto [p, d] : llvm::enumerate(dims)) {
      if (isInert(d)) {
        dropped = true;
      } else if (isUnitAxisPiece(d)) {
        units.push_back(d);
        unitOld.push_back(static_cast<int64_t>(p));
      } else {
        rest.push_back(d);
        restOld.push_back(static_cast<int64_t>(p));
      }
    }
    const bool moved =
        !units.empty() && unitOld.front() != static_cast<int64_t>(rest.size());
    if (dropped || moved) {
      SmallVector<int64_t> order(unitOld.size());
      for (size_t i = 0; i < order.size(); ++i) {
        order[i] = static_cast<int64_t>(i);
      }
      llvm::stable_sort(order, [&](int64_t x, int64_t y) {
        return units[x].getDim() < units[y].getDim();
      });
      SmallVector<int64_t> newPos(dims.size(), -1);
      SmallVector<DimAttr> out;
      for (auto [i, oldIdx] : llvm::enumerate(restOld)) {
        newPos[oldIdx] = static_cast<int64_t>(out.size());
        out.push_back(rest[i]);
      }
      for (int64_t i : order) {
        newPos[unitOld[i]] = static_cast<int64_t>(out.size());
        out.push_back(units[i]);
      }
      SmallVector<int64_t> keptRolls;
      for (size_t i = 0; i + 1 < rolls.size(); i += 2) {
        const int64_t from = rolls[i], by = rolls[i + 1];
        const bool fromGone = from >= 0 && newPos[from] < 0;
        const bool byGone = by >= 0 && newPos[by] < 0;
        if (fromGone || byGone) continue;
        keptRolls.push_back(from >= 0 ? newPos[from] : from);
        keptRolls.push_back(by >= 0 ? newPos[by] : by);
      }
      dims = std::move(out);
      rolls = std::move(keptRolls);
    }
  }
  const size_t ctLen = inferCtPrefixLen(dims, n);
  int64_t slotExtent = 1;
  for (size_t p = ctLen; p < dims.size(); ++p) slotExtent *= dims[p].getSize();
  if (slotExtent <= 0 || n % slotExtent != 0) return;
  const int64_t fill = n / slotExtent;
  if (fill <= 1) return;
  dims.insert(dims.begin() + ctLen,
              DimAttr::get(ctx, /*dim=*/-2, fill, /*stride=*/1));
  // Piece arguments at or past the insertion shift right.
  for (int64_t& encoded : rolls) {
    if (encoded >= static_cast<int64_t>(ctLen)) ++encoded;
  }
}

LayoutAttr LayoutAttr::getCanonical(MLIRContext* context,
                                    ArrayRef<DimAttr> dims, int64_t n,
                                    ArrayRef<int64_t> rolls) {
  SmallVector<DimAttr> dimVec(dims.begin(), dims.end());
  SmallVector<int64_t> rollVec(rolls.begin(), rolls.end());
  canonicalizeLayoutDims(context, dimVec, n, rollVec);
  SmallVector<Attribute> attrs(dimVec.begin(), dimVec.end());
  return get(context, ArrayAttr::get(context, attrs), n,
             DenseI64ArrayAttr::get(context, rollVec));
}

LayoutAttr LayoutAttr::get(MLIRContext* context, ArrayAttr dims, int64_t n) {
  SmallVector<DimAttr> dimVec;
  dimVec.reserve(dims.size());
  for (Attribute attr : dims) dimVec.push_back(cast<DimAttr>(attr));
  return getCanonical(context, dimVec, n);
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
