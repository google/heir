// The heir-translate executable in the python wheel: a thin launcher that runs
// heir-translate from libHEIRPythonCAPI.so (which holds all compiler code)
// instead of linking another copy of the compiler.

#include "lib/Bindings/Python/HEIRCAPI.h"

int main(int argc, char** argv) { return heirTranslateMain(argc, argv); }
