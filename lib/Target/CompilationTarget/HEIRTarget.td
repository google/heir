class CompilationTarget<string name> {
  string backendName = name;
  int bootstrapLevelsConsumed = 0;
}

def OpenFHE : CompilationTarget<"openfhe"> {
  let bootstrapLevelsConsumed = 3;
}

def Lattigo : CompilationTarget<"lattigo"> {
  let bootstrapLevelsConsumed = 1;
}