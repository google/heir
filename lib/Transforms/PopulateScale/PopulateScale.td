#ifndef LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_
#define LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_

include "mlir/Pass/PassBase.td"

def PopulateScaleBGV : Pass<"populate-scale-bgv"> {
  let summary = "Validate the HE circuit against a given noise model";
  let description = [{

  }];
}

def PopulateScaleCKKS : Pass<"populate-scale-ckks"> {
  let summary = "Validate the HE circuit against a given noise model";
  let description = [{

  }];

  let options = [
    Option<"beforeMulIncludeFirstMul", "before-mul-include-first-mul", "bool",
           /*default=*/"false", "Modulus switching before each multiplication, including the first multiplication (default to false)">,
  ];
}


#endif  // LIB_TRANSFORMS_POPULATESCALE_POPULATESCALE_TD_
