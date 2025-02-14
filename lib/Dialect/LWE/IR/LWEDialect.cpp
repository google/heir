#include "lib/Dialect/LWE/IR/LWEDialect.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/STLFunctionalExtras.h"   // from @llvm-project
#include "llvm/include/llvm/ADT/TypeSwitch.h"            // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"           // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Diagnostics.h"            // from @llvm-project
#include "mlir/include/mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"          // from @llvm-project

// Generated definitions
#include "lib/Dialect/LWE/IR/LWEDialect.cpp.inc"
#include "lib/Dialect/LWE/IR/LWEEnums.cpp.inc"
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OpImplementation.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
#define GET_OP_CLASSES
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"

namespace mlir {
namespace heir {
namespace lwe {

class LWEOpAsmDialectInterface : public OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  void getRingAttrAliasSuffix(polynomial::RingAttr ring,
                              raw_ostream& os) const {
    auto rnsType = dyn_cast<rns::RNSType>(ring.getCoefficientType());
    if (rnsType) {
      auto level = rnsType.getBasisTypes().size() - 1;
      os << "_L" << level;
    }
  }

  void getRLWEParamsAttrAliasSuffix(RLWEParamsAttr rlweParams,
                                    raw_ostream& os) const {
    auto dimension = rlweParams.getDimension();
    auto ring = rlweParams.getRing();
    getRingAttrAliasSuffix(ring, os);
    if (dimension != 2) {
      os << "_D" << dimension;
    }
  }

  void getCiphertextSpaceAttrAliasSuffix(CiphertextSpaceAttr ciphertextSpace,
                                         raw_ostream& os) const {
    auto ring = ciphertextSpace.getRing();
    getRingAttrAliasSuffix(ring, os);
    auto size = ciphertextSpace.getSize();
    if (size != 2) {
      os << "_D" << size;
    }
  }

  AliasResult getAlias(Type type, raw_ostream& os) const override {
    auto res =
        llvm::TypeSwitch<Type, AliasResult>(type)
            .Case<NewLWECiphertextType>([&](auto& type) {
              os << "ct";
              getCiphertextSpaceAttrAliasSuffix(type.getCiphertextSpace(), os);
              return AliasResult::FinalAlias;
            })
            .Case<NewLWEPlaintextType>([&](auto& type) {
              os << "pt";
              getRingAttrAliasSuffix(type.getPlaintextSpace().getRing(), os);
              return AliasResult::FinalAlias;
            })
            .Case<NewLWESecretKeyType>([&](auto& type) {
              os << "skey";
              getRingAttrAliasSuffix(type.getRing(), os);
              return AliasResult::FinalAlias;
            })
            .Case<NewLWEPublicKeyType>([&](auto& type) {
              os << "pkey";
              getRingAttrAliasSuffix(type.getRing(), os);
              return AliasResult::FinalAlias;
            })
            .Default([&](Type) { return AliasResult::NoAlias; });
    return res;
  }

  AliasResult getAlias(Attribute attr, raw_ostream& os) const override {
    auto res =
        llvm::TypeSwitch<Attribute, AliasResult>(attr)
            .Case<RLWEParamsAttr>([&](auto rlweParams) {
              os << "rlwe_params";
              getRLWEParamsAttrAliasSuffix(rlweParams, os);
              return AliasResult::FinalAlias;
            })
            .Case<BitFieldEncodingAttr>([&](auto bitFieldEncoding) {
              os << "bit_field_encoding";
              return AliasResult::FinalAlias;
            })
            .Case<UnspecifiedBitFieldEncodingAttr>(
                [&](auto unspecifiedBitFieldEncoding) {
                  os << "unspecified_bit_field_encoding";
                  return AliasResult::FinalAlias;
                })
            .Case<InverseCanonicalEncodingAttr>(
                [&](auto inverseCanonicalEncoding) {
                  os << "inverse_canonical_encoding";
                  return AliasResult::FinalAlias;
                })
            .Case<ModulusChainAttr>([&](auto modulusChain) {
              os << "modulus_chain";
              auto size = modulusChain.getElements().size();
              os << "_L" << size - 1;
              auto current = modulusChain.getCurrent();
              os << "_C" << current;
              return AliasResult::FinalAlias;
            })
            .Case<CiphertextSpaceAttr>([&](auto ciphertextSpace) {
              os << "ciphertext_space";
              getCiphertextSpaceAttrAliasSuffix(ciphertextSpace, os);
              return AliasResult::FinalAlias;
            })
            .Case<PlaintextSpaceAttr>([&](auto plaintextSpace) {
              os << "plaintext_space";
              getRingAttrAliasSuffix(plaintextSpace.getRing(), os);
              return AliasResult::FinalAlias;
            })
            .Case<KeyAttr>([&](auto key) {
              os << "key";
              return AliasResult::FinalAlias;
            })
            .Case<FullCRTPackingEncodingAttr>([&](auto fullCRTPackingEncoding) {
              os << "full_crt_packing_encoding";
              return AliasResult::FinalAlias;
            })
            .Default([&](Attribute) { return AliasResult::NoAlias; });
    return res;
  }
};

void LWEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/LWE/IR/LWEAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/LWE/IR/LWETypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/LWE/IR/LWEOps.cpp.inc"
      >();

  addInterface<LWEOpAsmDialectInterface>();
}

LogicalResult RMulOp::verify() {
  auto x = getLhs().getType();
  auto y = getRhs().getType();
  if (x.getCiphertextSpace().getSize() != y.getCiphertextSpace().getSize()) {
    return emitOpError() << "input dimensions do not match";
  }
  auto out = getOutput().getType();
  if (out.getCiphertextSpace().getSize() !=
      y.getCiphertextSpace().getSize() + x.getCiphertextSpace().getSize() - 1) {
    return emitOpError() << "output.dim == x.dim + y.dim - 1 does not hold";
  }
  return success();
}

LogicalResult RAddOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RAddOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  // NOT using FHEHelpers.h here because cyclic dependency
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::NewLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim = std::max(x.getCiphertextSpace().getSize(),
                         y.getCiphertextSpace().getSize());
  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain()));
  return success();
}

LogicalResult RSubOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RSubOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  // NOT using FHEHelpers.h here because cyclic dependency
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::NewLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim = std::max(x.getCiphertextSpace().getSize(),
                         y.getCiphertextSpace().getSize());
  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain()));
  return success();
}

LogicalResult RMulOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location>, RMulOp::Adaptor adaptor,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  // NOT using FHEHelpers.h here because cyclic dependency
  auto x = cast<lwe::NewLWECiphertextType>(adaptor.getLhs().getType());
  auto y = cast<lwe::NewLWECiphertextType>(adaptor.getRhs().getType());
  auto newDim =
      x.getCiphertextSpace().getSize() + y.getCiphertextSpace().getSize() - 1;
  inferredReturnTypes.push_back(lwe::NewLWECiphertextType::get(
      ctx, x.getApplicationData(), x.getPlaintextSpace(),
      lwe::CiphertextSpaceAttr::get(ctx, x.getCiphertextSpace().getRing(),
                                    x.getCiphertextSpace().getEncryptionType(),
                                    newDim),
      x.getKey(), x.getModulusChain()));
  return success();
}

LogicalResult BitFieldEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  if (!elementType.isSignlessInteger()) {
    return emitError() << "Tensors with a bit_field_encoding must have "
                       << "signless integer element type, but found "
                       << elementType;
  }

  unsigned plaintextBitwidth = elementType.getIntOrFloatBitWidth();
  unsigned cleartextBitwidth = getCleartextBitwidth();
  if (plaintextBitwidth < cleartextBitwidth)
    return emitError() << "The tensor element type's bitwidth "
                       << plaintextBitwidth
                       << " is too small to store the cleartext, "
                       << "which has bit width " << cleartextBitwidth << "";

  auto cleartextStart = getCleartextStart();
  if (cleartextStart < 0 || cleartextStart >= plaintextBitwidth)
    return emitError() << "Attribute's cleartext starting bit index ("
                       << cleartextStart << ") is outside the legal range [0, "
                       << plaintextBitwidth - 1 << "]";

  // It may be worth adding some sort of warning notification if the attribute
  // allocates no bits for noise, since this would be effectively useless for
  // FHE.
  return success();
}

LogicalResult UnspecifiedBitFieldEncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
  if (!elementType.isSignlessInteger()) {
    return emitError() << "Tensors with a bit_field_encoding must have "
                       << "signless integer element type, but found "
                       << elementType;
  }

  unsigned plaintextBitwidth = elementType.getIntOrFloatBitWidth();
  unsigned cleartextBitwidth = getCleartextBitwidth();
  if (plaintextBitwidth < cleartextBitwidth)
    return emitError() << "The tensor element type's bitwidth "
                       << plaintextBitwidth
                       << " is too small to store the cleartext, "
                       << "which has bit width " << cleartextBitwidth << "";

  return success();
}

LogicalResult TrivialEncryptOp::verify() {
  auto paramsAttr = this->getParamsAttr();
  auto outParamsAttr = this->getOutput().getType().getLweParams();

  if (paramsAttr != outParamsAttr) {
    return this->emitOpError()
           << "lwe_params attr must match on the op and "
              "the output type, but found op attr "
           << paramsAttr << " and output type attr " << outParamsAttr;
  }

  return success();
}

LogicalResult ReinterpretUnderlyingTypeOp::verify() {
  auto inputType = getInput().getType();
  auto outputType = getOutput().getType();
  if (inputType.getPlaintextSpace() != outputType.getPlaintextSpace() ||
      inputType.getCiphertextSpace() != outputType.getCiphertextSpace() ||
      inputType.getKey() != outputType.getKey() ||
      inputType.getModulusChain() != outputType.getModulusChain()) {
    return emitOpError()
           << "the only allowed difference in the input and output are in the "
              "application_data field, but found input type "
           << inputType << " and output type " << outputType;
  }

  return success();
}

// Verification for RLWE_EncryptOp
LogicalResult RLWEEncryptOp::verify() {
  Type keyType = getKey().getType();
  auto keyRing =
      llvm::TypeSwitch<Type, mlir::heir::polynomial::RingAttr>(keyType)
          .Case<lwe::NewLWEPublicKeyType, lwe::NewLWESecretKeyType>(
              [](auto key) { return key.getRing(); })
          .Default([](Type) {
            llvm_unreachable("impossible by type constraints");
            return nullptr;
          });

  auto outputRing = getOutput().getType().getCiphertextSpace().getRing();
  if (outputRing != keyRing) {
    return emitOpError() << "RLWEEncryptOp input ring do not match. Key ring: "
                         << keyRing
                         << ". Output ciphertext ring: " << outputRing << ".";
  }
  return success();
}

LogicalResult ApplicationDataAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::Type messageType, Attribute overflow) {
  if (!mlir::isa<PreserveOverflowAttr, NoOverflowAttr>(overflow)) {
    return emitError() << "overflow must be either preserve_overflow or "
                       << "no_overflow, but found " << overflow << "\n";
  }

  return success();
}

LogicalResult PlaintextSpaceAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    mlir::heir::polynomial::RingAttr ring, Attribute encoding) {
  if (mlir::isa<FullCRTPackingEncodingAttr>(encoding)) {
    // For full CRT packing, the ring must be of the form x^n + 1 and the
    // modulus must be 1 mod n.
    auto polyMod = ring.getPolynomialModulus();
    auto poly = polyMod.getPolynomial();
    auto polyTerms = poly.getTerms();
    if (polyTerms.size() != 2) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    const auto& constantTerm = polyTerms[0];
    const auto& constantCoeff = constantTerm.getCoefficient();
    if (!(constantTerm.getExponent().isZero() && constantCoeff.isOne() &&
          polyTerms[1].getCoefficient().isOne())) {
      return emitError() << "polynomial modulus must be of the form x^n + 1, "
                         << "but found " << polyMod << "\n";
    }
    // Check that the modulus is 1 mod n.
    auto modCoeffTy =
        llvm::dyn_cast<mod_arith::ModArithType>(ring.getCoefficientType());
    if (modCoeffTy) {
      APInt modulus = modCoeffTy.getModulus().getValue();
      unsigned n = poly.getDegree();
      if (!modulus.urem(APInt(modulus.getBitWidth(), n)).isOne()) {
        return emitError()
               << "modulus must be 1 mod n for full CRT packing, mod = "
               << modulus.getZExtValue() << " n = " << n << "\n";
      }
    }
  }

  return success();
}

LogicalResult NewLWECiphertextType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::heir::lwe::ApplicationDataAttr, mlir::heir::lwe::PlaintextSpaceAttr,
    mlir::heir::lwe::CiphertextSpaceAttr ciphertextSpace,
    mlir::heir::lwe::KeyAttr keyAttr,
    mlir::heir::lwe::ModulusChainAttr modulusChain) {
  if (keyAttr.getSlotIndex() != 0 && (ciphertextSpace.getSize() != 2)) {
    return emitError() << "a ciphertext with nontrivial slot rotation must "
                          "have size 2, but found size "
                       << ciphertextSpace.getSize();
  }
  if (auto rnsType = mlir::dyn_cast<rns::RNSType>(
          ciphertextSpace.getRing().getCoefficientType())) {
    if (rnsType.getBasisTypes().size() - 1 != modulusChain.getCurrent()) {
      return emitError() << "the level in the ciphertext ring "
                            "must match the modulus chain's current, but found "
                         << rnsType.getBasisTypes().size() - 1 << " and "
                         << modulusChain.getCurrent();
    }
  }
  return success();
}

// Verify Encoding and Type match
LogicalResult verifyEncodingAndTypeMatch(mlir::Type type,
                                         mlir::Attribute encoding) {
  // En/Decode Ops only allow IntegerOrFloatLike (-> assert not if)
  assert(getElementTypeOrSelf(type).isIntOrFloat() &&
         "Encoding Ops only allow IntegerOrFloatLike types");

  // Verification conditions for each encoding we have:

  if (isa<BitFieldEncodingAttr>(encoding)) {
    // Bit field encodings only work on (scalar/individual) integer
    return success(type.isInteger());
  }

  if (isa<UnspecifiedBitFieldEncodingAttr>(encoding)) {
    // same as BitFieldEncoding
    return success(type.isInteger());
  }

  // New LWE Encoding Attr

  if (isa<FullCRTPackingEncodingAttr>(encoding)) {
    // also supports lists of integers and scalars via replication
    return success(getElementTypeOrSelf(type).isInteger());
  }

  if (isa<InverseCanonicalEncodingAttr>(encoding)) {
    // CKKS-style Encoding should support everything
    // (ints via cast to float/double, scalars via replication)
    return success();
  }

  // This code should never be hit unless we added an encoding and forgot to
  // update this function. Assert(false) for DEBUG, return failure for NDEBUG.
  encoding.dump();
  assert(false && "Encoding not handled in encode/decode verifier.");
  return failure();
}

LogicalResult EncodeOp::verify() {
  return verifyEncodingAndTypeMatch(getInput().getType(), getEncoding());
}

LogicalResult RLWEEncodeOp::verify() {
  return verifyEncodingAndTypeMatch(getInput().getType(), getEncoding());
}

LogicalResult RLWEDecodeOp::verify() {
  return verifyEncodingAndTypeMatch(getResult().getType(), getEncoding());
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
