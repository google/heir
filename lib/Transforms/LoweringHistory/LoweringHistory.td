#ifndef LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_TD_
#define LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_TD_

include "mlir/Pass/PassBase.td"

def RecordLoweringHistory : Pass<"record-lowering-history"> {
  let summary = "Record operation names and a compilation stage in locations";
  let description = [{
    Wrap each operation's location with its current operation name and a named
    checkpoint. Subsequent rewrites that preserve locations retain these origins.
    Insert this pass at selected pipeline boundaries and use
    `--explain-lowering-history` to add readable history notes to errors.

    Checkpoints record the IR observed at a stage; they do not claim that every
    operation was rewritten by the preceding pass. Multiple origins are retained
    when a transformation fuses locations. Transformations that discard locations
    also discard this history. Use `--mlir-print-debuginfo` when serializing IR.

    Each checkpoint adds one location node per distinct operation-name/location
    pair. Prefer a few meaningful boundaries over recording every pass.
  }];
  let options = [Option<"stage", "stage", "std::string", "\"\"",
                        "Name of this checkpoint (required)">];
}

def ExplainLoweringHistory : Pass<"explain-lowering-history"> {
  let summary = "Attach recorded lowering-history notes to subsequent errors";
  let description = [{
    Install a diagnostic handler for the current MLIR context. Run once before
    the passes to debug. Errors retain their original message and source location;
    notes describe recorded checkpoints, most recent first. This also works on
    history read from a previous compiler process. The handler does not intercept
    or suppress errors. Missing history does not change diagnostics.
  }];
  let options = [Option<"maxNotes", "max-notes", "unsigned", "8",
                        "Maximum number of history entries per error (plus an omission note)">];
}

#endif  // LIB_TRANSFORMS_LOWERINGHISTORY_LOWERINGHISTORY_TD_
