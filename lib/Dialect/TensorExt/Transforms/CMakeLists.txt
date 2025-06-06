set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TensorExt)
add_public_tablegen_target(HEIRTensorExtPassesIncGen)

set(LLVM_TARGET_DEFINITIONS InsertRotate.td)
mlir_tablegen(InsertRotate.cpp.inc -gen-rewriters -name TensorExt)
add_public_tablegen_target(HEIRTensorExtInsertRotateIncGen)

add_mlir_library(HEIRTensorExtTransforms
    CollapseInsertionChains.cpp
    InsertRotate.cpp
    RotateAndReduce.cpp

    DEPENDS
    HEIRTensorExtPassesIncGen
    HEIRTensorExtInsertRotateIncGen

    LINK_LIBS PUBLIC
    HEIRTensorExt
    MLIRIR
    MLIRPass
    MLIRTransformUtils
)
