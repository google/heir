#include "lib/Dialect/Rotom/IR/RotomAttributes.h"

#include <cstddef>
#include <cstdint>
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

static int64_t computeImplicitFrontGap(ArrayRef<DimAttr> dims, int64_t n) {
  int64_t nRem = n;
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    DimAttr d = *it;
    if (d.isGap()) return 1;
    if (nRem <= 1) break;
    const int64_t sz = d.getSize();
    if (sz <= 0) return 1;
    if (sz <= nRem && nRem % sz == 0) nRem /= sz;
  }
  return nRem;
}

// Preprocesses a layout (`dims`, slot count `n`) into the `LayoutData`
// descriptor used to emit ciphertext addresses; also the validity check
// behind `LayoutAttr::verify`.
//
// On success `LayoutData` describes, in packing order: `pieces`, each tagged
// Traversal, Replication, or Gap with `pieceIndex` into its per-kind dim list;
// for each traversal piece the mixed-radix digit (i / pieceDivBy) mod
// pieceModBy read from tensor index i, so an axis's pieces share one domain
// variable; and `ctPrefixLen`, the ciphertext/slot split (with an implicit
// front gap when packed content leaves slots unfilled). See `LayoutData` in
// the header for exact field semantics.
//
// Fails on a malformed layout (unrecognized dim or non-positive `n`) or a
// multi-piece axis that is not a valid mixed-radix decomposition: sorted by
// stride, divisors must be the cumulative products of the lower extents and
// extents must multiply to the axis's full extent.
static FailureOr<LayoutData> preprocessLayoutData(ArrayAttr dims, int64_t n,
                                                  MLIRContext* ctx) {
  LayoutData data;
  data.n = n;
  if (data.n <= 0) return failure();

  llvm::DenseMap<int64_t, int64_t> traversalIndexForDim;
  data.originalDims.reserve(dims.size());
  for (Attribute a : dims) {
    auto d = dyn_cast<DimAttr>(a);
    if (!d) return failure();
    data.originalDims.push_back(d);
    if (d.isGap()) {
      data.pieceIndex.push_back(static_cast<int64_t>(data.gapDims.size()));
      data.gapDims.push_back(d);
      data.pieces.push_back(LayoutPieceKind::Gap);
      data.pieceDivBy.push_back(1);
      continue;
    }
    if (d.isReplicate()) {
      data.pieceIndex.push_back(
          static_cast<int64_t>(data.replicationDims.size()));
      data.replicationDims.push_back(d);
      data.pieces.push_back(LayoutPieceKind::Replication);
      data.pieceDivBy.push_back(1);
      continue;
    }
    if (d.getDim() >= 0) {
      auto [it, inserted] = traversalIndexForDim.try_emplace(
          d.getDim(), static_cast<int64_t>(data.traversalDims.size()));
      if (inserted) data.traversalDims.push_back(d);
      data.pieceIndex.push_back(it->second);
      data.pieces.push_back(LayoutPieceKind::Traversal);
      data.pieceDivBy.push_back(d.getStride());
      continue;
    }
    return failure();
  }

  // Canonicalize the deduped traversal dims to ascending dim id. The dedup
  // above collects them in first-appearance order, but consumers read the
  // list positionally as tensor dims -- most importantly the ISL lowering's
  // domain variables -- so a layout whose pieces lead with a later dim (e.g.
  // column-major [1:.][0:.]) must not leak that order into LayoutData.
  SmallVector<DimAttr> sortedTraversalDims = data.traversalDims;
  llvm::sort(sortedTraversalDims,
             [](DimAttr a, DimAttr b) { return a.getDim() < b.getDim(); });
  llvm::DenseMap<int64_t, int64_t> sortedIndexForDim;
  for (size_t ti = 0; ti < sortedTraversalDims.size(); ++ti) {
    sortedIndexForDim[sortedTraversalDims[ti].getDim()] =
        static_cast<int64_t>(ti);
  }
  data.traversalDims = std::move(sortedTraversalDims);
  for (size_t p = 0; p < data.pieces.size(); ++p) {
    if (data.pieces[p] == LayoutPieceKind::Traversal) {
      data.pieceIndex[p] = sortedIndexForDim[data.originalDims[p].getDim()];
    }
  }

  data.pieceModBy.assign(data.pieces.size(), 0);

  // Count pieces per tensor dim and the dim's full extent.
  llvm::DenseMap<int64_t, int64_t> pieceCount;
  llvm::DenseMap<int64_t, int64_t> dimFullExtent;
  for (size_t p = 0; p < data.pieces.size(); ++p) {
    if (data.pieces[p] != LayoutPieceKind::Traversal) continue;
    const int64_t dim = data.originalDims[p].getDim();
    ++pieceCount[dim];
    auto [it, inserted] = dimFullExtent.try_emplace(dim, 1);
    it->second *= data.originalDims[p].getSize();
  }

  for (size_t p = 0; p < data.pieces.size(); ++p) {
    if (data.pieces[p] != LayoutPieceKind::Traversal) continue;
    if (pieceCount[data.originalDims[p].getDim()] == 1) {
      // A whole dim packed as one piece keeps the legacy behavior: the stride
      // is not an address weight here, so ignore it -- digit == i.
      data.pieceDivBy[p] = 1;
      data.pieceModBy[p] = 0;
    } else {
      // An explicit mixed-radix split piece: digit = (i / stride) mod extent.
      // The modulus is suppressed (0) for the most-significant piece, whose
      // digit i / stride is already below its extent, to keep the emitted
      // address compact.
      const int64_t extent = data.originalDims[p].getSize();
      const int64_t full = dimFullExtent[data.originalDims[p].getDim()];
      data.pieceModBy[p] = (data.pieceDivBy[p] * extent < full) ? extent : 0;
    }
  }

  // Each multi-piece tensor dim must be a valid mixed-radix decomposition:
  // sorted by stride, the divisors are the cumulative products of the lower
  // extents (1, e0, e0*e1, ...), and the extents multiply to the full extent.
  for (auto& [dim, count] : pieceCount) {
    if (count == 1) continue;
    SmallVector<std::pair<int64_t, int64_t>> parts;  // (stride, extent)
    for (size_t p = 0; p < data.pieces.size(); ++p) {
      if (data.pieces[p] == LayoutPieceKind::Traversal &&
          data.originalDims[p].getDim() == dim) {
        parts.push_back({data.pieceDivBy[p], data.originalDims[p].getSize()});
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

  // Represent each traversal dim's domain variable by its full extent: the
  // deduped entry may be just the first of several mixed-radix pieces, but the
  // emitter derives the domain bound and each piece's effective extent
  // (full / divBy) from this entry. (Stride 1: it is not the address weight.)
  for (size_t ti = 0; ti < data.traversalDims.size(); ++ti) {
    const int64_t dim = data.traversalDims[ti].getDim();
    data.traversalDims[ti] =
        DimAttr::get(ctx, dim, dimFullExtent[dim], /*stride=*/1);
  }

  data.ctPrefixLen =
      static_cast<int64_t>(inferCtPrefixLen(data.originalDims, data.n));

  const int64_t implicitFrontGapSize =
      computeImplicitFrontGap(data.originalDims, data.n);
  if (implicitFrontGapSize > 1) {
    const int64_t gapIdx = static_cast<int64_t>(data.gapDims.size());
    data.gapDims.push_back(DimAttr::get(ctx, /*dim=*/-2,
                                        /*size=*/implicitFrontGapSize,
                                        /*stride=*/1));
    data.pieces.insert(data.pieces.begin() + data.ctPrefixLen,
                       LayoutPieceKind::Gap);
    data.pieceIndex.insert(data.pieceIndex.begin() + data.ctPrefixLen, gapIdx);
    data.pieceDivBy.insert(data.pieceDivBy.begin() + data.ctPrefixLen, 1);
    data.pieceModBy.insert(data.pieceModBy.begin() + data.ctPrefixLen, 0);
  }

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
  return failure(parser.parseColon() || parser.parseInteger(size) ||
                 parser.parseColon() || parser.parseInteger(stride) ||
                 parser.parseRSquare());
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

static ParseResult parseLayoutDims(AsmParser& parser,
                                   SmallVector<Attribute>& dims) {
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
    return parser.parseRSquare();
  }
}

static ParseResult parseLayoutRolls(AsmParser& parser,
                                    SmallVector<int64_t>& rolls,
                                    SmallVector<int64_t>& scales) {
  if (parser.parseLSquare()) return failure();
  if (succeeded(parser.parseOptionalRSquare())) return success();

  while (true) {
    if (succeeded(parser.parseOptionalLParen())) {
      int64_t from;
      int64_t to;
      int64_t scale = 1;
      if (parser.parseInteger(from) || parser.parseComma() ||
          parser.parseInteger(to))
        return failure();
      if (succeeded(parser.parseOptionalComma()) &&
          parser.parseInteger(scale)) {
        return failure();
      }
      if (parser.parseRParen()) return failure();
      rolls.push_back(from);
      rolls.push_back(to);
      scales.push_back(scale);
    } else {
      int64_t value;
      if (parser.parseInteger(value)) return failure();
      rolls.push_back(value);
      if (rolls.size() % 2 == 0) scales.push_back(1);
    }

    if (succeeded(parser.parseOptionalComma())) continue;
    return parser.parseRSquare();
  }
}

// The canonical scale storage: null when every scale is 1, so scale-free
// layouts unique to the same attribute whether or not scales were spelled.
static DenseI64ArrayAttr canonicalRollScales(MLIRContext* ctx,
                                             ArrayRef<int64_t> scales) {
  if (llvm::all_of(scales, [](int64_t s) { return s == 1; })) return {};
  return DenseI64ArrayAttr::get(ctx, scales);
}

static LogicalResult verifyLayoutRolls(
    ArrayAttr dims, DenseI64ArrayAttr rolls, DenseI64ArrayAttr rollScales,
    function_ref<InFlightDiagnostic()> emitError) {
  const bool noRolls = !rolls || rolls.empty();
  if (rollScales && !rollScales.empty()) {
    ArrayRef<int64_t> scales = rollScales.asArrayRef();
    if (noRolls || scales.size() * 2 != rolls.asArrayRef().size()) {
      return emitError() << "rollScales must hold one scale per roll pair";
    }
    if (llvm::any_of(scales, [](int64_t s) { return s == 0; })) {
      return emitError() << "roll scales must be non-zero";
    }
    if (llvm::all_of(scales, [](int64_t s) { return s == 1; })) {
      return emitError() << "rollScales must be omitted when every scale is 1";
    }
  }
  if (noRolls) return success();
  ArrayRef<int64_t> r = rolls.asArrayRef();
  if (r.size() % 2 != 0) {
    return emitError() << "rolls must contain an even number of integers "
                          "(pairs of dim indices)";
  }

  for (size_t i = 0; i < r.size(); i += 2) {
    const int64_t ti = r[i];
    const int64_t tj = r[i + 1];
    if (ti == tj) {
      return emitError() << "each roll must use two distinct dim indices";
    }
    if (ti < 0 || tj < 0 || ti >= static_cast<int64_t>(dims.size()) ||
        tj >= static_cast<int64_t>(dims.size())) {
      return emitError() << "roll dim index out of bounds for dims list";
    }
    auto di = dyn_cast<DimAttr>(dims[ti]);
    auto dj = dyn_cast<DimAttr>(dims[tj]);
    if (!di || !dj) {
      return emitError() << "roll indices must refer to #rotom.dim entries";
    }
    // The extents need not match: roll(i, j) rewrites dims[i]'s index to
    // (i_i - i_j) mod size(dims[i]), well-defined for any partner extent (a
    // smaller partner covers a prefix of the rotations, a larger one wraps).
    // The rolled (from) dim must be a traversal dim -- it is the index
    // expression being rewritten. The roll-by (second) dim may be any kind:
    // rolling by a replication or gap dim shifts by that dim's block index,
    // so each block holds a distinct cyclic rotation of the rolled dim -- the
    // layout materializes every rotation and alignment becomes block
    // selection. (A rolled-by gap thus claims its blocks, unlike a plain gap.)
    if (di.isGap() || di.isReplicate()) {
      return emitError() << "the rolled dim must be a traversal dim (dim >= 0)";
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
    DenseI64ArrayAttr scalesAttr = getRollScales();
    ArrayRef<int64_t> scales =
        scalesAttr ? scalesAttr.asArrayRef() : ArrayRef<int64_t>{};
    printer << ", rolls = [";
    for (size_t i = 0; i < values.size(); i += 2) {
      if (i != 0) printer << ", ";
      printer << "(" << values[i] << ", " << values[i + 1];
      const int64_t scale = i / 2 < scales.size() ? scales[i / 2] : 1;
      if (scale != 1) printer << ", " << scale;
      printer << ")";
    }
    printer << "]";
  }

  printer << ", dims = [";
  llvm::interleaveComma(getDims(), printer, [&](Attribute attr) {
    printDimTriple(printer, cast<DimAttr>(attr));
  });
  printer << "]>";
}

Attribute LayoutAttr::parse(AsmParser& parser, Type type) {
  int64_t n;
  SmallVector<int64_t> rolls;
  SmallVector<int64_t> scales;
  SmallVector<Attribute> dims;

  if (parser.parseLess()) return {};

  if (succeeded(parser.parseOptionalKeyword("n"))) {
    if (parser.parseEqual() || parser.parseInteger(n) || parser.parseComma()) {
      return {};
    }

    if (succeeded(parser.parseOptionalKeyword("rolls"))) {
      if (parser.parseEqual() ||
          failed(parseLayoutRolls(parser, rolls, scales)) ||
          parser.parseComma()) {
        return {};
      }
    }

    if (parser.parseKeyword("dims") || parser.parseEqual() ||
        failed(parseLayoutDims(parser, dims)) || parser.parseGreater()) {
      return {};
    }
  } else if (succeeded(parser.parseOptionalKeyword("dims"))) {
    if (parser.parseEqual() || failed(parseLayoutDims(parser, dims)) ||
        parser.parseComma() || parser.parseKeyword("n") ||
        parser.parseEqual() || parser.parseInteger(n)) {
      return {};
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseKeyword("rolls") || parser.parseEqual() ||
          failed(parseLayoutRolls(parser, rolls, scales))) {
        return {};
      }
    }

    if (parser.parseGreater()) return {};
  } else {
    parser.emitError(parser.getNameLoc())
        << "expected `n` or `dims` in rotom layout";
    return {};
  }

  MLIRContext* context = parser.getContext();
  return LayoutAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, context,
      ArrayAttr::get(context, dims), n, DenseI64ArrayAttr::get(context, rolls),
      canonicalRollScales(context, scales));
}

FailureOr<LayoutData> preprocessLayoutAttr(LayoutAttr layout) {
  return preprocessLayoutData(layout.getDims(), layout.getN(),
                              layout.getContext());
}

LogicalResult LayoutAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayAttr dims, int64_t n,
                                 DenseI64ArrayAttr rolls,
                                 DenseI64ArrayAttr rollScales) {
  if (n <= 0) {
    return emitError() << "`n` must be > 0, got " << n;
  }
  auto preprocessed = preprocessLayoutData(dims, n, dims.getContext());
  if (failed(preprocessed)) {
    return emitError() << "`dims` must be an array of `#rotom.dim<...>`";
  }

  if (failed(verifyLayoutRolls(dims, rolls, rollScales, emitError))) {
    return failure();
  }

  MLIRContext* ctx = dims.getContext();
  std::vector<DimAttr> ctDims;
  std::vector<DimAttr> slotDims;

  int64_t nRem = n;
  for (auto it = preprocessed->originalDims.rbegin();
       it != preprocessed->originalDims.rend(); ++it) {
    DimAttr d = *it;
    const int64_t size = d.getSize();

    if (nRem <= 1) {
      ctDims.insert(ctDims.begin(), d);
      continue;
    }

    // Size > nRem: split into ct and slot dims
    if (size > nRem) {
      if (size % nRem != 0) {
        return emitError() << "dim size " << size
                           << " must be divisible by remaining slot capacity "
                           << nRem;
      }

      slotDims.insert(slotDims.begin(),
                      DimAttr::get(ctx, d.getDim(), nRem, /*stride=*/1));
      ctDims.insert(ctDims.begin(), DimAttr::get(ctx, d.getDim(), size / nRem,
                                                 /*stride=*/nRem));
      nRem /= size;
      continue;
    }

    // Size == nRem: add to slot dims
    if (size == nRem) {
      slotDims.insert(slotDims.begin(), d);
      nRem /= size;
      continue;
    }

    // Size divides nRem: add to slot dims
    if (nRem % size == 0) {
      slotDims.insert(slotDims.begin(), d);
      nRem /= size;
      continue;
    }

    // Size does not divide nRem (e.g. odd input channels): keep in ctDims
    // so slotDims can remain pow2 and nRem is unchanged.
    ctDims.insert(ctDims.begin(), d);
  }

  // If there is remaining slot capacity, insert a gap dim at the front.
  if (nRem > 1) {
    slotDims.insert(slotDims.begin(),
                    DimAttr::get(ctx, /*dim=*/-2, /*size=*/nRem,
                                 /*stride=*/1));
  }

  // Remove gap dims from ctDims.
  std::vector<DimAttr> ctDimsFiltered;
  ctDimsFiltered.reserve(ctDims.size());
  for (DimAttr d : ctDims) {
    if (!d.isGap()) ctDimsFiltered.push_back(d);
  }
  ctDims.swap(ctDimsFiltered);

  // Enforce the Rotom invariant: slot-dim sizes/strides must be powers of 2.
  for (DimAttr d : slotDims) {
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getSize()))) {
      return emitError() << "slot dim size must be a power of two, got "
                         << d.getSize();
    }
    if (!llvm::isPowerOf2_64(static_cast<uint64_t>(d.getStride()))) {
      return emitError() << "slot dim stride must be a power of two, got "
                         << d.getStride();
    }
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
                                  layoutAttr.getN(), layoutAttr.getRolls(),
                                  layoutAttr.getRollScales())))
      return failure();
  }
  return success();
}

LayoutAttr LayoutAttr::get(MLIRContext* context, ArrayAttr dims, int64_t n) {
  return get(context, dims, n,
             DenseI64ArrayAttr::get(context, ArrayRef<int64_t>{}),
             DenseI64ArrayAttr());
}

LayoutAttr LayoutAttr::get(MLIRContext* context, ArrayAttr dims, int64_t n,
                           DenseI64ArrayAttr rolls) {
  return get(context, dims, n, rolls, DenseI64ArrayAttr());
}

}  // namespace rotom
}  // namespace heir
}  // namespace mlir
