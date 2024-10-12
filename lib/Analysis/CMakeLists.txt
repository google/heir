# Convenience target that all Analysis targets should be dependencies of
add_library(HEIRAnalysis INTERFACE)

add_subdirectory(MulDepthAnalysis)
add_subdirectory(OptimizeRelinearizationAnalysis)
add_subdirectory(RotationAnalysis)
add_subdirectory(SecretnessAnalysis)
add_subdirectory(SelectVariableNames)
add_subdirectory(TargetSlotAnalysis)
