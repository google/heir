#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"  // from @googletest
#include "gtest/gtest.h"  // from @googletest
#include "lib/Dialect/Comb/IR/CombDialect.h"
#include "lib/Dialect/Comb/IR/CombOps.h"
#include "lib/Transforms/YosysOptimizer/LUTImporter.h"
#include "lib/Transforms/YosysOptimizer/RTLILImporter.h"
#include "llvm/include/llvm/ADT/STLExtras.h"             // from @llvm-project
#include "llvm/include/llvm/Support/Path.h"              // from @llvm-project
#include "mlir/include/mlir//IR/Location.h"              // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project

// Block clang-format from reordering
// clang-format off
#include "tools/cpp/runfiles/runfiles.h"  // from @bazel_tools
#include "kernel/log.h"  // from @at_clifford_yosys
#include "kernel/rtlil.h"  // from @at_clifford_yosys
#include "kernel/yosys.h"  // from @at_clifford_yosys
// clang-format on

namespace mlir::heir {
namespace {

using bazel::tools::cpp::runfiles::Runfiles;
using ::testing::Test;

static constexpr std::string_view kWorkspaceDir = "heir";

class LUTImporterTestFixture : public Test {
 protected:
  void SetUp() override {
    context.loadDialect<heir::comb::CombDialect, arith::ArithDialect,
                        func::FuncDialect, tensor::TensorDialect>();
    module_ = ModuleOp::create(UnknownLoc::get(&context));
    Yosys::yosys_setup();
  }

  func::FuncOp runImporter(const std::string& rtlil) {
    std::unique_ptr<Runfiles> runfiles(Runfiles::CreateForTest());
    SmallString<128> workspaceRelativePath;
    llvm::sys::path::append(workspaceRelativePath, kWorkspaceDir, rtlil);
    Yosys::run_pass("read_rtlil " +
                    runfiles->Rlocation(workspaceRelativePath.str().str()));
    Yosys::run_pass("proc; hierarchy -generate lut* o:Y i:P i:*;");

    // Get topological ordering.
    std::stringstream cellOrder;
    Yosys::log_streams.push_back(&cellOrder);
    Yosys::run_pass("torder -stop i P*;");
    Yosys::log_streams.clear();

    LUTImporter lutImporter = LUTImporter(&context);

    auto topologicalOrder = getTopologicalOrder(cellOrder);
    Yosys::RTLIL::Design* design = Yosys::yosys_get_design();
    auto func = lutImporter.importModule(design->top_module(), topologicalOrder,
                                         std::nullopt);
    module_->push_back(func);
    return func;
  }

  void TearDown() override { Yosys::run_pass("delete"); }

  MLIRContext context;

  OwningOpRef<ModuleOp> module_;
};

// Note that we cannot lower truth tables to LLVM, so we must assert the IR
// rather than executing the code.
TEST_F(LUTImporterTestFixture, AddOneLUT3) {
  std::vector<uint8_t> expectedLuts = {6, 120, 128, 6, 120, 128,
                                       6, 120, 128, 6, 1};

  auto func =
      runImporter("lib/Transforms/YosysOptimizer/tests/add_one_lut3.rtlil");

  auto funcType = func.getFunctionType();
  EXPECT_EQ(funcType.getNumInputs(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getInput(0)).getNumElements(), 8);
  EXPECT_EQ(funcType.getNumResults(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getResult(0)).getNumElements(), 8);

  auto combOps = func.getOps<comb::TruthTableOp>().begin();
  for (size_t i = 0; i < expectedLuts.size(); i++) {
    auto lutValue = (*combOps++).getLookupTable().getValue();
    EXPECT_THAT(lutValue, APInt(lutValue.getBitWidth(), expectedLuts[i]));
  }
}

TEST_F(LUTImporterTestFixture, AddOneLUT5) {
  auto func =
      runImporter("lib/Transforms/YosysOptimizer/tests/add_one_lut5.rtlil");

  auto funcType = func.getFunctionType();
  EXPECT_EQ(funcType.getNumInputs(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getInput(0)).getNumElements(), 8);
  EXPECT_EQ(funcType.getNumResults(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getResult(0)).getNumElements(), 8);

  auto combOps = func.getOps<comb::TruthTableOp>();
  for (auto combOp : combOps) {
    auto lutValue = combOp.getLookupTable().getValue();
    EXPECT_EQ(lutValue.getBitWidth(), 32);
  }
}

// This test doubles the input, which simply connects the output wire to the
// first bits of the input and a constant.
// comb.concat inputs are written in MSB to LSB ordering. See
// https://circt.llvm.org/docs/Dialects/Comb/RationaleComb/#endianness-operand-ordering-and-internal-representation
TEST_F(LUTImporterTestFixture, DoubleInput) {
  auto func =
      runImporter("lib/Transforms/YosysOptimizer/tests/double_input.rtlil");

  auto funcType = func.getFunctionType();
  EXPECT_EQ(funcType.getNumInputs(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getInput(0)).getNumElements(), 8);
  EXPECT_EQ(funcType.getNumResults(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getResult(0)).getNumElements(), 8);

  auto returnOp = *func.getOps<func::ReturnOp>().begin();
  auto fromElementsOp =
      returnOp.getOperands()[0].getDefiningOp<tensor::FromElementsOp>();
  ASSERT_TRUE(fromElementsOp);
  // Expect 8 values stored in the result tensor.
  EXPECT_EQ(fromElementsOp.getElements().size(), 8);

  // The 0th digit of the result must be zero.
  Value firstVal = *fromElementsOp.getElements().begin();
  arith::ConstantOp firstValue = firstVal.getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(firstValue);
  auto constVal = dyn_cast<IntegerAttr>(firstValue.getValue());
  ASSERT_TRUE(constVal);
  EXPECT_EQ(constVal.getInt(), 0);
}

TEST_F(LUTImporterTestFixture, MultipleInputs) {
  auto func =
      runImporter("lib/Transforms/YosysOptimizer/tests/multiple_inputs.rtlil");

  auto funcType = func.getFunctionType();
  EXPECT_EQ(funcType.getNumInputs(), 2);
  EXPECT_EQ(cast<TensorType>(funcType.getInput(0)).getNumElements(), 8);
  EXPECT_EQ(cast<TensorType>(funcType.getInput(1)).getNumElements(), 8);
  EXPECT_EQ(funcType.getNumResults(), 1);
  EXPECT_EQ(cast<TensorType>(funcType.getResult(0)).getNumElements(), 8);
}

}  // namespace
}  // namespace mlir::heir

// We use a custom main function here to avoid issues with Yosys' main driver.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
