#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lib/Target/OpenFhePke/Interpreter.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"          // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "src/core/include/lattice/hal/lat-backend.h"  // from @openfhe
#include "src/pke/include/cryptocontext-fwd.h"         // from @openfhe

using namespace lbcrypto;
using namespace mlir::heir::openfhe;
using CryptoContextT = CryptoContext<DCRTPoly>;
using floatvec = std::shared_ptr<std::vector<float>>;
using mlir::MLIRContext;
using mlir::ModuleOp;

std::pair<std::vector<float>, double> lenet_interpreter(
    const std::string& mlirSrc, const std::vector<float>& input) {
  // Load the MLIR module from a file
  MLIRContext context;
  initContext(context);
  mlir::OwningOpRef<ModuleOp> module = parse(&context, mlirSrc);
  Interpreter interpreter(module.get());

  std::cout << "Generating crypto context" << std::endl;
  TypedCppValue ccInitial =
      interpreter.interpret("lenet__generate_crypto_context", {})[0];

  auto keyPair = std::get<CryptoContextT>(ccInitial.value)->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  std::vector<TypedCppValue> args = {ccInitial, TypedCppValue(secretKey)};
  std::cout << "Configuring crypto context" << std::endl;
  TypedCppValue cc = std::move(
      interpreter.interpret("lenet__configure_crypto_context", args)[0]);

  std::cout << "Encrypting input" << std::endl;
  TypedCppValue arg0Enc = interpreter.interpret(
      "lenet__encrypt__arg0",
      {cc, TypedCppValue(input), TypedCppValue(publicKey)})[0];

  std::cout << "Running module" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  TypedCppValue outputEncrypted =
      interpreter.interpret("lenet", {cc, arg0Enc})[0];
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Decrypting output" << std::endl;
  std::vector<TypedCppValue> actualVal =
      interpreter.interpret("lenet__decrypt__result0",
                            {cc, outputEncrypted, TypedCppValue(secretKey)});

  auto resultVec = std::get<floatvec>(actualVal[0].value);
  double duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  return {*resultVec, duration};
}
