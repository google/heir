add_heir_pass(MemrefToArith)

add_mlir_library(HEIRMemrefGlobalReplace
    PARTIAL_SOURCES_INTENDED
    MemrefGlobalReplace.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTransformUtils
    MLIRSupport
)
target_link_libraries(HEIRTransforms INTERFACE HEIRMemrefGlobalReplace)

add_mlir_library(HEIRExpandCopy
    PARTIAL_SOURCES_INTENDED
    ExpandCopy.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRExpandCopy)

add_mlir_library(HEIRExtractLoopBody
    PARTIAL_SOURCES_INTENDED
    ExtractLoopBody.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRExtractLoopBody)

add_mlir_library(HEIRUnrollAndForward
    PARTIAL_SOURCES_INTENDED
    UnrollAndForward.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRAffineAnalysis
    MLIRAffineDialect
    MLIRAffineUtils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTransformUtils
)
target_link_libraries(HEIRTransforms INTERFACE HEIRUnrollAndForward)

add_mlir_library(HEIRMemrefToArithRegistration
    PARTIAL_SOURCES_INTENDED
    PassRegistration.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    HEIRExpandCopy
    HEIRExtractLoopBody
    HEIRMemrefGlobalReplace
    HEIRUnrollAndForward
)
target_link_libraries(HEIRTransforms INTERFACE HEIRMemrefToArithRegistration)

add_mlir_library(HEIRMemrefToArithUtils
    PARTIAL_SOURCES_INTENDED
    Utils.cpp

    DEPENDS
    HEIRMemrefToArithIncGen

    LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineAnalysis
    MLIRAffineUtils
    LLVMSupport
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
target_link_libraries(HEIRTransforms INTERFACE HEIRMemrefToArithUtils)
