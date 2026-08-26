// RUN: heir-opt --convert-to-emitc=filter-dialects=math %s | FileCheck %s

// CHECK: emitc.include <"cmath">
// CHECK: func.func @square_root
// CHECK: %[[ROOT:.*]] = emitc.call_opaque "std::sqrt"(%arg0) : (f32) -> f32
// CHECK: return %[[ROOT]] : f32
func.func @square_root(%arg0: f32) -> f32 {
  %0 = math.sqrt %arg0 : f32
  return %0 : f32
}
