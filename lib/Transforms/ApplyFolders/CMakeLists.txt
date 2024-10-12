add_heir_pass(ApplyFolders)

add_mlir_library(HEIRApplyFolders
    ApplyFolders.cpp

    DEPENDS
    HEIRApplyFoldersIncGen

    LINK_LIBS PUBLIC
    HEIRModArith
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRArithDialect
    MLIRSupport
    MLIRDialect
    MLIRIR
)
target_link_libraries(HEIRTransforms INTERFACE HEIRApplyFolders)
