#ifndef LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_
#define LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_

include "mlir/Pass/PassBase.td"

def PopulateScale : Pass<"populate-scale"> {
  let summary = "Validate the HE circuit against a given noise model";
  let description = [{

  }];
}

#endif  // LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_
