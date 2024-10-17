# Convenience target that all utils target should be dependencies of
add_library(HEIRUtils INTERFACE)

add_subdirectory(ConversionUtils)
add_subdirectory(DRR)
add_subdirectory(Graph)
add_subdirectory(TargetUtils)
