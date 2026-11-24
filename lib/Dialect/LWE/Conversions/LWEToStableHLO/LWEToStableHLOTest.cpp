#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/LWE/Conversions/LWEToStableHLO/LWEToStableHLO.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "llvm/include/llvm/ADT/APInt.h"        // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncDialect.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"          // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/DialectRegistry.h"     // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"         // from @llvm-project
#include "mlir/include/mlir/Parser/Parser.h"          // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"           // from @stablehlo
#include "stablehlo/reference/Api.h"                  // from @stablehlo
#include "stablehlo/reference/Configuration.h"        // from @stablehlo

namespace mlir::heir::lwe {
namespace {

TEST(LWEToStableHLOTest,
     EvaluatesRAddOnCanonicalRank5TensorWithBoundaryWraparound) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect, lwe::LWEDialect,
                  mod_arith::ModArithDialect, polynomial::PolynomialDialect,
                  rns::RNSDialect, stablehlo::StablehloDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  constexpr const char* kIr = R"mlir(
    !Z2147473409 = !mod_arith.int<2147473409 : i64>
    !Z4294967291 = !mod_arith.int<4294967291 : i64>
    !rns_ty = !rns.rns<!Z2147473409, !Z4294967291>
    #encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
    #key = #lwe.key<>
    #ring = #polynomial.ring<coefficientType = !rns_ty, polynomialModulus = <1 + x**4>>
    #ct_space = #lwe.ciphertext_space<ring = #ring, encryption_type = ckks>
    !ct = !lwe.lwe_ciphertext<encoding = #encoding, ciphertext_space = #ct_space, key = #key>

    module {
      func.func @main(%lhs: !ct, %rhs: !ct) -> !ct {
        %sum = lwe.radd %lhs, %rhs : (!ct, !ct) -> !ct
        return %sum : !ct
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kIr, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(createLWEToStableHLO());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  constexpr uint64_t kQ0 = 2147473409ULL;
  constexpr uint64_t kQ1 = 4294967291ULL;  // Largest 32-bit prime (< 2^32)
  // Shape: (batch=1, size=2, r=2, c=2, num_moduli=2) -> 16 ui32 elements
  const std::vector<uint32_t> lhsRaw = {
      0U,          0U,           // 0 + 0
      2147473408U, 4294967290U,  // (q - 1) + 1 -> 0
      2147473408U, 4294967290U,  // (q - 1) + (q - 1) -> q - 2
      1000000000U, 3000000000U,  // mid-range wrap on limb 1
      123456789U,  987654321U,   // no wrap
      2147473400U, 4294967200U,  // wrap on both
      1U,          2U,           // small values
      500000000U,  2147483648U,  // > 2^31 on limb 1
  };
  const std::vector<uint32_t> rhsRaw = {
      0U,          0U,          1U,          1U,
      2147473408U, 4294967290U, 1000000000U, 2000000000U,
      111111111U,  222222222U,  100U,        200U,
      3U,          4U,          500000000U,  2147483648U,
  };

  auto ui32Ty = IntegerType::get(&context, 32, IntegerType::Unsigned);
  auto tensorTy = RankedTensorType::get({1, 2, 2, 2, 2}, ui32Ty);
  SmallVector<APInt> lhsAp, rhsAp;
  for (size_t i = 0; i < lhsRaw.size(); ++i) {
    lhsAp.emplace_back(32, lhsRaw[i], /*isSigned=*/false);
    rhsAp.emplace_back(32, rhsRaw[i], /*isSigned=*/false);
  }
  auto lhsAttr = DenseElementsAttr::get(tensorTy, lhsAp);
  auto rhsAttr = DenseElementsAttr::get(tensorTy, rhsAp);

  stablehlo::InterpreterConfiguration config;
  FailureOr<SmallVector<DenseElementsAttr>> results =
      stablehlo::evalModule(*module, {lhsAttr, rhsAttr}, config);
  ASSERT_TRUE(succeeded(results));
  ASSERT_EQ(results->size(), 1U);

  auto outValues = (*results)[0].getValues<APInt>();
  ASSERT_EQ(outValues.size(), lhsRaw.size());
  for (size_t i = 0; i < lhsRaw.size(); ++i) {
    uint64_t q = (i % 2 == 0) ? kQ0 : kQ1;
    uint64_t expected =
        (static_cast<uint64_t>(lhsRaw[i]) + static_cast<uint64_t>(rhsRaw[i])) %
        q;
    EXPECT_EQ(outValues[i].getZExtValue(), expected) << "at flat index " << i;
  }
}

}  // namespace
}  // namespace mlir::heir::lwe
