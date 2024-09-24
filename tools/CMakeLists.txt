get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

function(make_heirtx_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE
        ${dialect_libs} ${conversion_libs}

        MLIRAutoHogImporter
        MLIRJaxiteTarget
        MLIRMetadataEmitter
        MLIROpenfheTarget
        MLIRTfheRustTarget
        MLIRTfheRustBoolTarget
        MLIRVerilogTarget
        MLIRSupport
        MLIRTranslateLib
        MLIRCGGI

        )
endfunction()

function(make_heirlsp_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE
        ${dialect_libs} ${conversion_libs}

        MLIRHEIRInterfaces
        MLIRHEIRUtils
        MLIRLspServerLib
        MLIROptLib
        MLIRSCFDialect
        MLIRMemRefDialect
        MLIRComb
        MLIRCGGI
        MLIRBGV
        MLIRLWE
        MLIRSecret
        MLIRModArithDialect
        MLIRTensorExt
        MLIRTensorDialect
        MLIRHEIRAnalysisROT
        MLIRAffineDialect
        MLIRJaxiteTarget
        MLIRMetadataEmitter
        MLIRSupport
        MLIRTranslateLib
        MLIRVerilogTarget
        MLIROptLib MLIROpenfhe MLIRLWE MLIROpenfheTarget MLIRSecret MLIRVerilogTarget MLIRHEIRTargetUtils
        MLIRTfheRust MLIRTfheRustBool MLIRHEIRAnalysisSVN
        MLIRTfheRustTarget
        MLIRTfheRustBoolTarget
        MLIRHEIRTargetUtils
        )
endfunction()

function(make_heiropt_exec  TARGET_NAME SOURCES)
        add_llvm_executable(${TARGET_NAME} PARTIAL_SOURCES_INTENDED ${SOURCES})
        llvm_update_compile_flags(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE
        ${dialect_libs} ${conversion_libs}

        MLIRJaxiteTarget
        MLIRModArithDialect
        MLIRSupport
        MLIRTranslateLib
        MLIRVerilogTarget
        MLIROpenfhe
        MLIRLWE
        MLIROpenfheTarget
        MLIRSecret
        MLIRVerilogTarget
        MLIRHEIRTargetUtils
        MLIRTfheRust
        MLIRTfheRustBool
        MLIRHEIRAnalysisSVN
        MLIRTfheRustTarget
        MLIRTfheRustBoolTarget
        MLIRHEIRTargetUtils

        MLIRBGV
        MLIRCGGI
        MLIRComb
        MLIRJaxite
        MLIRLWE
        MLIROpenfhe
        MLIRRNS
        MLIRSecret
        MLIRTensorExt
        MLIRTfheRust
        MLIRTfheRustBool

        MLIRSupport
        MLIRAffineDialect
        MLIRAffineToStandard
        MLIRAffineTransforms
        MLIRFuncAllExtensions
        MLIRFuncDialect
        MLIRArithDialect
        MLIRArithToLLVM
        MLIRArithTransforms
        MLIRBufferizationToMemRef
        MLIRBufferizationTransforms
        MLIRControlFlowToLLVM
        MLIRConvertToLLVMPass
        MLIRFuncDialect
        MLIRFuncToLLVM
        MLIRFuncTransforms
        MLIRIndexToLLVM
        MLIRLLVMDialect
        MLIRLinalgTransforms
        MLIRMemRefDialect
        MLIRMemrefToArithMemrefGlobalReplace
        MLIRMemRefToLLVM
        MLIRMemRefTransforms
        MLIROptLib
        MLIRPass
        MLIRPolynomialDialect
        MLIRReconcileUnrealizedCasts
        MLIRSCFDialect
        MLIRSCFToControlFlow
        MLIRTensorToLinalg
        MLIRTensorTransforms
        MLIRTosaDialect
        MLIRTosaToArith
        MLIRTosaToLinalg
        MLIRTosaToTensor
        MLIRTransforms
        MLIRHeirConvertIfToSelect
        MLIRHeiElementwiseToAffine
        MLIRHeirForwardStoreToLoad
        MLIRHeirFullLoopUnroll
        MLIRHeirLinalgCanonicalizations
        MLIRHeirSecretize
        MLIRHeirStraightLineVectorizer

        MLIRPolynomialTransforms
        MLIRHeirPolynomialNTTRewrites

        MLIRMemrefToArithUtils
        MLIRMemrefToArithRegistration
        MLIRMemrefToArithUnrollAndForward
        MLIRMemrefToArithExtractLoopBody
        MLIRMemrefToArithExpandCopy
        MLIRMemrefToArithMemrefGlobalReplace
        MLIRBGVOpenfheTransforms
        MLIRCKKSOpenfheTransforms
        MLIRRlweToOpenfheTransforms
        MLIRLWEToPolynomial
        MLIRHeirConvertSecretForToStaticFor
        MLIRHeirConvertSecretWhileToStaticFor

        MLIRRandom
        MLIRCGGIToJaxiteTransforms
        MLIRCGGIToTfheRustTransforms
        MLIRCKKS
)
add_dependencies(${TARGET_NAME} MLIRPolynomialPassesIncGen)
add_dependencies(${TARGET_NAME} MLIRPolynomialNTTRewritePassesIncGen)
add_dependencies(${TARGET_NAME} MLIRTosaToSecretArith)
add_dependencies(${TARGET_NAME} MLIRLinalgToTensorExt)
endfunction()

message(STATUS "Configuring heir-translate")
make_heirtx_exec(heir-translate heir-translate.cpp)

message(STATUS "Configuring heir-lsp")
make_heirlsp_exec(heir-lsp heir-lsp.cpp)

message(STATUS "Configuring heir-opt")
make_heiropt_exec(heir-opt heir-opt.cpp)
