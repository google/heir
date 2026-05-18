#ifndef LIB_TARGET_LATTIGO_LATTIGOTRANSLATEREGISTRATION_H_
#define LIB_TARGET_LATTIGO_LATTIGOTRANSLATEREGISTRATION_H_

namespace mlir {
namespace heir {
namespace lattigo {

void registerTranslateOptions();

void registerToLattigoTranslation();

void registerToLattigoPreprocessingTranslation();

void registerToLattigoPreprocessedTranslation();

void registerToLattigoDebugTranslation();

}  // namespace lattigo
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_LATTIGO_LATTIGOTRANSLATEREGISTRATION_H_
