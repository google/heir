#include "lib/Dialect/Polynomial/Transforms/AttachNTTRoots.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/IR/Builders.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace polynomial {

#define GEN_PASS_DEF_ATTACHNTTROOTS
#include "lib/Dialect/Polynomial/Transforms/Passes.h.inc"

std::optional<mod_arith::ModArithAttr> getModArithRoot(
    MLIRContext* context, mod_arith::ModArithType maTy, int degree,
    int useInvRoot) {
  std::optional<APInt> root =
      findPrimitive2nthRoot(maTy.getModulus().getValue(), degree);
  if (root) {
    if (useInvRoot) {
      APInt cmod = maTy.getModulus().getValue();
      root = multiplicativeInverse(root->zext(cmod.getBitWidth()), cmod);
    }
    return mod_arith::ModArithAttr::get(
        context, maTy, IntegerAttr::get(maTy.getModulus().getType(), *root));
  }
  return std::nullopt;
}

LogicalResult attachNTTRootsGeneric(
    Operation* op, PatternRewriter& rewriter, Value input,
    std::optional<PrimitiveRootAttr> primRootAttr, bool forwardNTT) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);
  bool useInvRoot = !forwardNTT;
  std::string msg;

  PolynomialType polyTy;
  if (auto p = dyn_cast<PolynomialType>(input.getType())) {
    polyTy = p;
  } else if (auto rt = dyn_cast<RankedTensorType>(input.getType())) {
    polyTy = dyn_cast<PolynomialType>(rt.getElementType());
  }

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
  IntegerAttr cycIndex = b.getI64IntegerAttr(2 * degree);
  Type coeffType = ring.getCoefficientType();
  PrimitiveRootAttr rootAttr;

  if (auto modArithType = dyn_cast<mod_arith::ModArithType>(coeffType)) {
    std::optional<mod_arith::ModArithAttr> root =
        getModArithRoot(context, modArithType, degree, useInvRoot);
    if (!root) {
      return rewriter.notifyMatchFailure(
          op, "Unable to compute primitive root for ModArith type");
    }
    rootAttr = PrimitiveRootAttr::get(context, *root, cycIndex);
  } else if (auto rnsType = dyn_cast<rns::RNSType>(coeffType)) {
    SmallVector<Attribute> rootValues;
    rootValues.reserve(rnsType.getBasisTypes().size());
    for (Type basisType : rnsType.getBasisTypes()) {
      auto limbType = dyn_cast<mod_arith::ModArithType>(basisType);
      if (!limbType) {
        llvm::raw_string_ostream(msg)
            << "Expected ModArith component in RNS type; got " << basisType;
        return rewriter.notifyMatchFailure(op, msg);
      }
      std::optional<mod_arith::ModArithAttr> root =
          getModArithRoot(context, limbType, degree, useInvRoot);
      if (!root) {
        return rewriter.notifyMatchFailure(
            op, "Unable to compute primitive root for RNS type");
      }
      rootValues.push_back(*root);
    }
    rns::RNSAttr rootValue = rns::RNSAttr::get(context, rootValues, rnsType);
    rootAttr = PrimitiveRootAttr::get(context, rootValue, cycIndex);
  } else {
    llvm::raw_string_ostream(msg)
        << "polyMulToNTT cannot create an NTT root for coefficient type "
        << coeffType;
    return rewriter.notifyMatchFailure(op, msg);
  }
  if (forwardNTT) {
    rewriter.replaceOp(op, NTTOp::create(b, input, rootAttr));
  } else {
    rewriter.replaceOp(op, INTTOp::create(b, input, rootAttr));
  }
  return success();
}

LogicalResult AttachNTTRootsPattern::matchAndRewrite(
    NTTOp op, PatternRewriter& rewriter) const {
  if (op.getRoot()) {
    return success();
  }
  return attachNTTRootsGeneric(op, rewriter, op.getInput(), op.getRoot(), true);
}

LogicalResult AttachINTTRootsPattern::matchAndRewrite(
    INTTOp op, PatternRewriter& rewriter) const {
  if (op.getRoot()) {
    return success();
  }
  return attachNTTRootsGeneric(op, rewriter, op.getInput(), op.getRoot(),
                               false);
}

struct AttachNTTRoots : impl::AttachNTTRootsBase<AttachNTTRoots> {
  using AttachNTTRootsBase::AttachNTTRootsBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<AttachNTTRootsPattern, AttachINTTRootsPattern>(context);

    (void)walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

}  // namespace polynomial
}  // namespace heir
}  // namespace mlir
