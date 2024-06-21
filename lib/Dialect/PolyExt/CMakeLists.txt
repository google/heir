add_subdirectory(IR)
add_mlir_dialect_library(MLIRPolyExt
        IR/PolyExtDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/IR

        DEPENDS
        MLIRPolyExtOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRSupport
        MLIRDialect
        MLIRIR
)
