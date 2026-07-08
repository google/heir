#include "lib/Dialect/Rotom/IR/RotomAttributes.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"           // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"          // from @llvm-project
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
// behind `LayoutAttr::verify`.
//
// `pieces` describes the traversal dimensions, replication dimensions, and
// gap dimensions. When lowering to ISL, a traversal piece maps to
// `(i / divBy) mod modBy`, where i is the index of the piece's tensor axis;
// replication and gap pieces map to their own ISL existential variables
// instead.
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
      // Split piece: digit = (i / stride) mod extent.
      const int64_t extent = piece.dim.getSize();
      const int64_t full = dimFullExtent[piece.dim.getDim()];
      piece.modBy = (piece.divBy * extent < full) ? extent : 0;
    }
  }

  // Each multi-piece tensor dim must be a valid mixed-radix decomposition:
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

// The dim position is spelled `R` for replication and `G` for gap (the
// readable forms, also how the printer emits them); the numeric ids -1 and
// -2 are still accepted.
static ParseResult parseDimTripleAfterLSquare(AsmParser& parser, int64_t& dim,
                                              int64_t& size, int64_t& stride) {
  if (succeeded(parser.parseOptionalKeyword("R"))) {
    dim = -1;
  } else if (succeeded(parser.parseOptionalKeyword("G"))) {
    dim = -2;
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

// One endpoint of a roll pair: a bare non-negative integer is a dims-list
// position (one piece); `axis N` names the whole tensor axis N.
static ParseResult parseRollEndpoint(AsmParser& parser, int64_t& encoded) {
  const bool isAxis = succeeded(parser.parseOptionalKeyword("axis"));
  int64_t value;
  if (parser.parseInteger(value)) return failure();
  if (value < 0) {
    return parser.emitError(parser.getNameLoc())
           << (isAxis ? "an axis roll endpoint must name a non-negative "
                        "tensor axis"
                      : "a piece roll endpoint must be a non-negative dims "
                        "position (spell a whole-axis endpoint as `axis N`)");
  }
  encoded = encodeRollEndpoint({isAxis, value});
  return success();
}

static ParseResult parseLayoutRolls(AsmParser& parser,
                                    SmallVector<int64_t>& rolls) {
  if (parser.parseLSquare()) return failure();
  if (succeeded(parser.parseOptionalRSquare())) return success();

  // Each entry is one complete roll pair, either a `(from, to)` tuple or two
  // bare endpoints `from, to`.
  while (true) {
    int64_t from;
    int64_t to;
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parseRollEndpoint(parser, from)) || parser.parseComma() ||
          failed(parseRollEndpoint(parser, to)) || parser.parseRParen())
        return failure();
    } else {
      if (failed(parseRollEndpoint(parser, from)) || parser.parseComma() ||
          failed(parseRollEndpoint(parser, to)))
        return failure();
    }
    rolls.push_back(from);
    rolls.push_back(to);

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
    return emitError() << "rolls must contain an even number of endpoints "
                          "(pairs)";
  }

  // Traversal pieces of a tensor axis: their count decides whether an `axis`
  // endpoint is legal (split axes only -- the piece spelling is canonical
  // when the axis is one piece), and their extent product is the modulus a
  // whole-axis rewrite reduces by.
  auto piecesOfAxis = [&](int64_t axis) {
    std::pair<int64_t, int64_t> countAndExtent{0, 1};
    for (Attribute a : dims) {
      auto d = dyn_cast<DimAttr>(a);
      if (d && !d.isGap() && !d.isReplicate() && d.getDim() == axis) {
        ++countAndExtent.first;
        countAndExtent.second *= d.getSize();
      }
    }
    return countAndExtent;
  };

  for (size_t i = 0; i < r.size(); i += 2) {
    const RollEndpoint from = decodeRollEndpoint(r[i]);
    const RollEndpoint by = decodeRollEndpoint(r[i + 1]);

    // Resolve each endpoint: the piece it names (null for axis endpoints)
    // and the tensor axis it reads or rewrites (sentinel for gap/replication
    // pieces).
    DimAttr fromPiece;
    DimAttr byPiece;
    int64_t fromAxis = 0;
    int64_t byAxis = 0;
    auto checkEndpoint = [&](const RollEndpoint& e, DimAttr& piece,
                             int64_t& axis) -> LogicalResult {
      if (e.isAxis) {
        auto [count, extent] = piecesOfAxis(e.index);
        (void)extent;
        if (count == 0) {
          return emitError() << "an axis roll endpoint must name a tensor "
                                "axis present in dims";
        }
        if (count == 1) {
          return emitError() << "an axis roll endpoint requires a split "
                                "axis; spell an unsplit axis's endpoint as "
                                "its piece position";
        }
        axis = e.index;
        return success();
      }
      if (e.index >= static_cast<int64_t>(dims.size())) {
        return emitError() << "roll piece endpoint out of bounds for dims "
                              "list";
      }
      piece = dyn_cast<DimAttr>(dims[e.index]);
      if (!piece) {
        return emitError() << "roll endpoints must refer to #rotom.dim "
                              "entries";
      }
      axis = piece.getDim();
      return success();
    };
    if (failed(checkEndpoint(from, fromPiece, fromAxis)) ||
        failed(checkEndpoint(by, byPiece, byAxis))) {
      return failure();
    }

    // The extents need not match: a roll rewrites the from index to
    // (idx - shift) mod extent(from), well-defined for any partner extent (a
    // smaller partner covers a prefix of the rotations, a larger one wraps).
    // FROM is the index expression being rewritten, so it must be a
    // traversal piece or a whole (traversal) axis. The by endpoint may be
    // any kind: rolling by a replication or gap piece shifts by that piece's
    // block index, so each block holds a distinct cyclic rotation of the
    // rolled index -- the layout materializes every rotation and alignment
    // becomes block selection. (A rolled-by gap thus claims its blocks,
    // unlike a plain gap.)
    if (!from.isAxis && (fromPiece.isGap() || fromPiece.isReplicate())) {
      return emitError() << "the rolled dim must be a traversal dim (dim >= "
                            "0)";
    }
    // A roll may not shift an index by itself. Piece endpoints must be
    // distinct positions (two pieces of one axis are distinct digits); an
    // axis endpoint overlaps every endpoint on the same axis, because a
    // whole-axis rewrite touches all of its digits.
    if (!from.isAxis && !by.isAxis && from.index == by.index) {
      return emitError() << "each roll must use two distinct endpoints";
    }
    if ((from.isAxis || by.isAxis) && fromAxis == byAxis) {
      return emitError() << "a roll may not shift an axis by one of its own "
                            "pieces";
    }
    // A rolled-by GAP claims one ciphertext block per gap index, each holding
    // a distinct rotation of the rolled index. If the gap is larger than the
    // rolled extent the rotations repeat (period = the from extent), claiming
    // blocks the conversion/kernel accounting was never audited for.
    // (Replication partners of larger extent are intended -- replicate-then-
    // roll -- so only gaps are bounded.)
    const int64_t fromExtent =
        from.isAxis ? piecesOfAxis(fromAxis).second : fromPiece.getSize();
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
  if (dim < -2) {
    return emitError() << "`dim` must be >= -2, got " << dim;
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
    auto printEndpoint = [&](int64_t encoded) {
      const RollEndpoint e = decodeRollEndpoint(encoded);
      if (e.isAxis) printer << "axis ";
      printer << e.index;
    };
    printer << ", rolls = [";
    for (size_t i = 0; i < values.size(); i += 2) {
      if (i != 0) printer << ", ";
      printer << "(";
      printEndpoint(values[i]);
      printer << ", ";
      printEndpoint(values[i + 1]);
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

SmallVector<RollSpec> getRollSpecs(LayoutAttr layout) {
  SmallVector<RollSpec> specs;
  DenseI64ArrayAttr rolls = layout.getRolls();
  if (!rolls) return specs;
  ArrayRef<int64_t> r = rolls.asArrayRef();

  for (size_t i = 0; i + 1 < r.size(); i += 2) {
    specs.push_back({decodeRollEndpoint(r[i]), decodeRollEndpoint(r[i + 1])});
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
                       << "); spell unused capacity as an explicit gap piece";
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
  const size_t ctLen = inferCtPrefixLen(dims, n);
  int64_t slotExtent = 1;
  for (size_t p = ctLen; p < dims.size(); ++p) slotExtent *= dims[p].getSize();
  if (slotExtent <= 0 || n % slotExtent != 0) return;
  const int64_t fill = n / slotExtent;
  if (fill <= 1) return;
  dims.insert(dims.begin() + ctLen,
              DimAttr::get(ctx, /*dim=*/-2, fill, /*stride=*/1));
  // Piece endpoints at or past the insertion shift right; axis endpoints
  // (encoded negative) name axes and do not move.
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
