add_subdirectory(IR)
add_mlir_dialect_library(MLIRComb
        IR/CombDialect.cpp
        IR/CombOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRCombIncGen
        MLIRCombOpsIncGen
        MLIRCombDialectEnumsIncGen
        MLIRCombTypesIncGen

        LINK_LIBS PUBLIC
        LLVMSupport
        MLIRHEIRInterfaces
        MLIRBytecodeOpInterface
        MLIRFunctionInterfaces
        MLIRSideEffectInterfaces
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
