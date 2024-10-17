add_mlir_library(HEIRSelectVariableNames
        SelectVariableNames.cpp

        LINK_LIBS PUBLIC
        MLIRIR
        LLVMSupport
)
target_link_libraries(HEIRAnalysis INTERFACE HEIRSelectVariableNames)
