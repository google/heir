# Convenience target that all Target libraries should be dependencies of
add_library(HEIRTarget INTERFACE)

add_subdirectory(Jaxite)
add_subdirectory(Metadata)
add_subdirectory(OpenFhePke)
add_subdirectory(TfheRust)
add_subdirectory(TfheRustBool)
add_subdirectory(Verilog)
