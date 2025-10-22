#ifndef LIB_TARGET_OPENFHEPKE_INTERPRETER_H_
#define LIB_TARGET_OPENFHEPKE_INTERPRETER_H_

#include <string>
#include <variant>
#include <vector>

#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "src/pke/include/openfhe.h"                     // from @openfhe

namespace mlir {
namespace heir {
namespace openfhe {

using CCParamsT = lbcrypto::CCParams<lbcrypto::CryptoContextCKKSRNS>;
using CiphertextT = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
using CryptoContextT = lbcrypto::CryptoContext<lbcrypto::DCRTPoly>;
using EvalKeyT = lbcrypto::EvalKey<lbcrypto::DCRTPoly>;
using PlaintextT = lbcrypto::Plaintext;
using PrivateKeyT = lbcrypto::PrivateKey<lbcrypto::DCRTPoly>;
using PublicKeyT = lbcrypto::PublicKey<lbcrypto::DCRTPoly>;
using FastRotPrecompT = std::shared_ptr<std::vector<lbcrypto::DCRTPoly>>;

struct TypedCppValue {
  using Variant =
      std::variant<std::monostate,                    // NULL_TY
                   bool,                              // BOOL
                   int,                               // INT
                   float,                             // FLOAT
                   double,                            // DOUBLE
                   std::vector<int>,                  // INT_VECTOR
                   std::vector<float>,                // FLOAT_VECTOR
                   std::vector<double>,               // DOUBLE_VECTOR
                   PlaintextT,                        // PLAINTEXT
                   std::vector<lbcrypto::Plaintext>,  // PLAINTEXT_VECTOR
                   CiphertextT,                       // CIPHERTEXT
                   std::vector<CiphertextT>,          // CIPHERTEXT_VECTOR
                   PublicKeyT,                        // PUBLIC_KEY
                   PrivateKeyT,                       // PRIVATE_KEY
                   EvalKeyT,                          // EVAL_KEY
                   CryptoContextT,                    // CRYPTO_CONTEXT
                   CCParamsT,                         // CRYPTO_CONTEXT_PARAMS
                   FastRotPrecompT                    // FAST_ROTATION_PRECOMP
                   >;

  Variant value;

  TypedCppValue() = default;
  TypedCppValue(bool v) : value(v) {}
  TypedCppValue(int v) : value(v) {}
  TypedCppValue(float v) : value(v) {}
  TypedCppValue(double v) : value(v) {}
  TypedCppValue(std::vector<int> v) : value(std::move(v)) {}
  TypedCppValue(std::vector<float> v) : value(std::move(v)) {}
  TypedCppValue(std::vector<double> v) : value(std::move(v)) {}
  TypedCppValue(PlaintextT v) : value(std::move(v)) {}
  TypedCppValue(std::vector<PlaintextT> v) : value(std::move(v)) {}
  TypedCppValue(CiphertextT v) : value(std::move(v)) {}
  TypedCppValue(std::vector<CiphertextT> v) : value(std::move(v)) {}
  TypedCppValue(PublicKeyT v) : value(std::move(v)) {}
  TypedCppValue(PrivateKeyT v) : value(std::move(v)) {}
  TypedCppValue(EvalKeyT v) : value(std::move(v)) {}
  TypedCppValue(CryptoContextT v) : value(std::move(v)) {}
  TypedCppValue(CCParamsT v) : value(std::move(v)) {}
  TypedCppValue(FastRotPrecompT v) : value(std::move(v)) {}
};

class Interpreter {
 public:
  Interpreter(ModuleOp module) : module(module) {}

  std::vector<TypedCppValue> interpret(const std::string& entryFunction,
                                       ArrayRef<TypedCppValue> inputValues);

  void visit(Operation* op);

  // Upstream ops
  void visit(arith::AddIOp op);
  void visit(arith::AndIOp op);
  void visit(arith::CmpIOp op);
  void visit(arith::ConstantOp op);
  void visit(arith::DivSIOp op);
  void visit(arith::ExtFOp op);
  void visit(arith::ExtSIOp op);
  void visit(arith::ExtUIOp op);
  void visit(arith::IndexCastOp op);
  void visit(arith::MulIOp op);
  void visit(arith::RemSIOp op);
  void visit(arith::SelectOp op);
  void visit(arith::SubIOp op);
  void visit(tensor::CollapseShapeOp op);
  void visit(tensor::ConcatOp op);
  void visit(tensor::EmptyOp op);
  void visit(tensor::ExpandShapeOp op);
  void visit(tensor::ExtractOp op);
  void visit(tensor::ExtractSliceOp op);
  void visit(tensor::FromElementsOp op);
  void visit(tensor::InsertOp op);
  void visit(tensor::InsertSliceOp op);
  void visit(tensor::SplatOp op);

  // SCF and Affine ops
  void visit(scf::IfOp op);
  void visit(scf::ForOp op);
  void visit(scf::YieldOp op);
  void visit(affine::AffineForOp op);
  void visit(affine::AffineYieldOp op);

  // OpenFHE ops
  void visit(AddOp op);
  void visit(AddPlainOp op);
  void visit(AutomorphOp op);
  void visit(BootstrapOp op);
  void visit(DecryptOp op);
  void visit(EncryptOp op);
  void visit(FastRotationOp op);
  void visit(FastRotationPrecomputeOp op);
  void visit(GenBootstrapKeyOp op);
  void visit(GenContextOp op);
  void visit(GenMulKeyOp op);
  void visit(GenParamsOp op);
  void visit(GenRotKeyOp op);
  void visit(KeySwitchOp op);
  void visit(LevelReduceOp op);
  void visit(MakeCKKSPackedPlaintextOp op);
  void visit(MakePackedPlaintextOp op);
  void visit(ModReduceOp op);
  void visit(MulConstOp op);
  void visit(MulNoRelinOp op);
  void visit(MulOp op);
  void visit(MulPlainOp op);
  void visit(NegateOp op);
  void visit(RelinOp op);
  void visit(RotOp op);
  void visit(SetupBootstrapOp op);
  void visit(SquareOp op);
  void visit(SubOp op);
  void visit(SubPlainOp op);

  // Other HEIR ops
  void visit(lwe::RLWEDecodeOp op);

  TypedCppValue applyBinop(Operation* op, const TypedCppValue& lhs,
                           const TypedCppValue& rhs,
                           std::function<int(int, int)> intFunc,
                           std::function<double(double, double)> doubleFunc);

  int getFlattenedTensorIndex(Value tensor, ValueRange indices);

 private:
  ModuleOp module;
  llvm::DenseMap<Value, TypedCppValue> env;
};

void initContext(MLIRContext& context);

OwningOpRef<ModuleOp> parse(MLIRContext* context, const std::string& mlirStr);

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_INTERPRETER_H_
