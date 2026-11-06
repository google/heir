/* Linker version script for libHEIRPythonCAPI.so (Linux). Keep in sync with
   libHEIRPythonCAPI.exp.

   Only export what the python extensions use: the MLIR and HEIR C APIs, and
   the C++ runtime. Everything else (MLIR, LLVM, HEIR) stays local, which keeps
   it out of the dynamic symbol table, lets the linker drop unused code, and
   keeps it from clashing with other copies loaded into the same process.

   The C++ runtime (libc++, libc++abi and the unwinder) is linked statically
   into this library, and the extensions use this copy rather than linking their
   own, so that there is a single C++ runtime in the process. This matters, e.g.,
   for exceptions: nanobind's exception translators are shared by all extensions
   (and registered by whichever loads first), and only recognize exception types
   from the same runtime. */
{
  global:
    mlir*;
    heir*;

    /* libc++ (std::...), including typeinfo, vtables and VTTs. */
    _ZNSt*;
    _ZNKSt*;
    _ZSt*;
    _ZTISt*;
    _ZTSSt*;
    _ZTVSt*;
    _ZTTSt*;
    _ZTINSt*;
    _ZTSNSt*;
    _ZTVNSt*;
    _ZTTNSt*;
    _ZTCNSt*;
    /* libc++abi and the unwinder. */
    _ZTIN10__cxxabiv1*;
    _ZTSN10__cxxabiv1*;
    _ZTVN10__cxxabiv1*;
    _Znw*;
    _Zna*;
    _Zdl*;
    _Zda*;
    __cxa_*;
    __dynamic_cast*;
    __gxx_personality_*;
    _Unwind_*;
  local:
    *;
};
