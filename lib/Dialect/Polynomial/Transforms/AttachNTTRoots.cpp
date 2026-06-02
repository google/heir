#include "lib/Dialect/Polynomial/Transforms/AttachNTTRoots.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "lib/Dialect/ModArith/IR/ModArithAttributes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialTypes.h"
#include "lib/Dialect/RNS/IR/RNSAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/MathUtils.h"
#include "llvm/include/llvm/Support/raw_ostream.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_ATTACHNTTROOTS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

// Helper to compute or fetch the limb-level root
std::optional<mod_arith::ModArithAttr> getModArithRoot(
    MLIRContext* context, RingAttr ringTy, mod_arith::ModArithType maTy,
    int degree, bool useInvRoot, RootCache& cache) {
  auto& limbCache = useInvRoot ? cache.inttRoots : cache.nttRoots;

  // Check cache
  if (limbCache.count(ringTy)) {
    return cast<mod_arith::ModArithAttr>(limbCache[ringTy]);
  }

  // Compute the forward root if needed
  // If !useInvRoot, this root doesn't exist, so we need to compute it
  // If useInvRoot, the inverse root isn't in the cache, so we need the
  // forward root to be in the cache for the last block
  if (!cache.nttRoots.count(ringTy)) {
    std::optional<APInt> nttRoot =
        findPrimitive2nthRoot(maTy.getModulus().getValue(), degree);
    if (!nttRoot) return std::nullopt;
    cache.nttRoots[ringTy] = mod_arith::ModArithAttr::get(
        context, maTy, IntegerAttr::get(maTy.getModulus().getType(), *nttRoot));
  }

  // derive iNTT root from cached NTT root. If we're here, the forward root is
  // already in the cache.
  if (useInvRoot) {
    APInt nttRoot = cast<mod_arith::ModArithAttr>(cache.nttRoots[ringTy])
                        .getValue()
                        .getValue();
    APInt cmod = maTy.getModulus().getValue();
    APInt inttRoot =
        multiplicativeInverse(nttRoot.zextOrTrunc(cmod.getBitWidth()), cmod);
    auto maRoot = mod_arith::ModArithAttr::get(
        context, maTy, IntegerAttr::get(maTy.getModulus().getType(), inttRoot));
    limbCache[ringTy] = maRoot;
  }

  return cast<mod_arith::ModArithAttr>(limbCache[ringTy]);
}

LogicalResult attachNTTRootsGeneric(Operation* op, PatternRewriter& rewriter,
                                    Value input, bool forwardNTT,
                                    RootCache& cache) {
  bool useInvRoot = !forwardNTT;
  std::string msg;

  PolynomialType polyTy =
      dyn_cast<PolynomialType>(getElementTypeOrSelf(input.getType()));

  if (!polyTy) {
    llvm::raw_string_ostream(msg) << "polyMulToNTT:getPrimitiveRootAttr "
                                     "expected polynomial-like type, got "
                                  << input.getType();
    return rewriter.notifyMatchFailure(op, msg);
  }

  RingAttr ring = polyTy.getRing();
  IntPolynomialAttr polyMod = ring.getPolynomialModulus();
  if (!polyMod) {
    llvm::raw_string_ostream(msg) << "polyMulToNTT cannot create an NTT root "
                                     "for ring without a polynomial modulus: "
                                  << ring;
    return rewriter.notifyMatchFailure(op, msg);
  }

  MLIRContext* context = op->getContext();
  uint64_t degree = polyMod.getPolynomial().getDegree();
  // The cyclotomic index is twice the modulus degree
  IntegerAttr cycIndex = rewriter.getI64IntegerAttr(2 * degree);
  Type coeffType = ring.getCoefficientType();
  PrimitiveRootAttr rootAttr;

  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(coeffType)) {
    std::optional<mod_arith::ModArithAttr> root =
        getModArithRoot(context, ring, modArithType, degree, useInvRoot, cache);
    if (!root) {
      return rewriter.notifyMatchFailure(
          op, "Unable to compute primitive root for ModArith type");
    }
    rootAttr = PrimitiveRootAttr::get(context, *root, cycIndex);
  } else if (auto rnsType = dyn_cast<rns::RNSType>(coeffType)) {
    rns::RNSAttr rootValue;
    auto& typeCache = useInvRoot ? cache.inttRoots : cache.nttRoots;
    if (typeCache.count(ring)) {
      rootValue = cast<rns::RNSAttr>(typeCache[ring]);
    } else {
      SmallVector<Attribute> rootValues;
      rootValues.reserve(rnsType.getBasisTypes().size());
      for (Type basisType : rnsType.getBasisTypes()) {
        auto limbType = dyn_cast<mod_arith::ModArithType>(basisType);
        if (!limbType) {
          llvm::raw_string_ostream(msg)
              << "Expected ModArith component in RNS type; got " << basisType;
          return rewriter.notifyMatchFailure(op, msg);
        }
        RingAttr limbRng = RingAttr::get(limbType, polyMod);
        std::optional<mod_arith::ModArithAttr> root = getModArithRoot(
            context, limbRng, limbType, degree, useInvRoot, cache);
        if (!root) {
          return rewriter.notifyMatchFailure(
              op, "Unable to compute primitive root for RNS type");
        }
        rootValues.push_back(*root);
      }
      rootValue = rns::RNSAttr::get(context, rootValues, rnsType);
      typeCache[ring] = rootValue;
    }
    rootAttr = PrimitiveRootAttr::get(context, rootValue, cycIndex);
  } else {
    llvm::raw_string_ostream(msg)
        << "polyMulToNTT cannot create an NTT root for coefficient type "
        << coeffType;
    return rewriter.notifyMatchFailure(op, msg);
  }
  rewriter.modifyOpInPlace(op, [&] { op->setAttr("root", rootAttr); });
  return success();
}

LogicalResult AttachNTTRootsPattern::matchAndRewrite(
    NTTOp op, PatternRewriter& rewriter) const {
  if (op.getRoot()) {
    return success();
  }
  return attachNTTRootsGeneric(op, rewriter, op.getInput(), true, cache);
}

LogicalResult AttachINTTRootsPattern::matchAndRewrite(
    INTTOp op, PatternRewriter& rewriter) const {
  if (op.getRoot()) {
    return success();
  }
  return attachNTTRootsGeneric(op, rewriter, op.getInput(), false, cache);
}

struct AttachNTTRoots : impl::AttachNTTRootsBase<AttachNTTRoots> {
  using AttachNTTRootsBase::AttachNTTRootsBase;

  RootCache cache;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<AttachNTTRootsPattern, AttachINTTRootsPattern>(context, cache);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
