set(LLVM_TARGET_DEFINITIONS MemrefToArith.td)
mlir_tablegen(MemrefToArith.h.inc -gen-pass-decls -name MemrefToArith)
add_public_tablegen_target(MLIRMemrefToArithPassIncGen)

add_mlir_library(MLIRMemrefToArithMemrefGlobalReplace
    PARTIAL_SOURCES_INTENDED
    MemrefGlobalReplace.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

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

add_mlir_library(MLIRMemrefToArithExpandCopy
    PARTIAL_SOURCES_INTENDED
    ExpandCopy.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

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

add_mlir_library(MLIRMemrefToArithExtractLoopBody
    PARTIAL_SOURCES_INTENDED
    ExtractLoopBody.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

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

add_mlir_library(MLIRMemrefToArithUnrollAndForward
    PARTIAL_SOURCES_INTENDED
    UnrollAndForward.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

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

add_mlir_library(MLIRMemrefToArithRegistration
    PARTIAL_SOURCES_INTENDED
    PassRegistration.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

    LINK_LIBS PUBLIC
    MLIRMemrefToArithExpandCopy
    MLIRMemrefToArithExtractLoopBody
    MLIRMemrefToArithMemrefGlobalReplace
    MLIRMemrefToArithUnrollAndForward
)

add_mlir_library(MLIRMemrefToArithUtils
    PARTIAL_SOURCES_INTENDED
    Utils.cpp

    DEPENDS
    MLIRMemrefToArithPassIncGen

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
