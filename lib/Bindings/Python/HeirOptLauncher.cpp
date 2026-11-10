// The heir-opt executable in the python wheel: a thin launcher that runs
// heir-opt from libHEIRPythonCAPI.so (which holds all compiler code) instead of
// linking another copy of the compiler.

#include "lib/Bindings/Python/HEIRCAPI.h"

int main(int argc, char** argv) { return heirOptMain(argc, argv); }
