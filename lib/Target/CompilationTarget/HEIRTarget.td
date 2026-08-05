#ifndef LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
#define LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_

class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
  int has_kernel_chebyshev = 0;
}

#endif // LIB_TARGET_COMPILATIONTARGET_HEIRTARGET_TD_
