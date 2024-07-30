add_subdirectory(IR)
add_mlir_dialect_library(MLIRRNS
        IR/RNSDialect.cpp
        IR/RNSTypes.cpp
        IR/RNSOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRRNSIncGen
        MLIRRNSTypeIncGen
        MLIRRNSOpsIncGen
        MLIRRNSTypeInterfaces

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRArithDialect
        MLIRSupport
        MLIRDialect
        MLIRIR
)
