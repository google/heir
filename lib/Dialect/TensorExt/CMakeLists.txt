add_subdirectory(IR)
add_subdirectory(Transforms)
add_mlir_dialect_library(MLIRTensorExt
        IR/TensorExtDialect.cpp
        IR/TensorExtOps.cpp
        Transforms/CollapseInsertionChains.cpp
        Transforms/InsertRotate.cpp
        Transforms/RotateAndReduce.cpp
        Transforms/AlignTensorSizes.cpp
        Transforms/Passes.h

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRTensorExtIncGen
        MLIRTensorExtOpsIncGen
        MLIRTensorExtAttributesIncGen
        MLIRTensorExtPassesIncGen
        MLIRTensorExtInsertRotateIncGen
        MLIRTensorExtCanonicalizationIncGen
        MLIRSecretIncGen

        LINK_LIBS PUBLIC
        MLIRSecret
        MLIRIR
        MLIRTensorDialect
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
