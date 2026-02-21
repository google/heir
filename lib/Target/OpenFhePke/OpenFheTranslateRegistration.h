
#ifndef LIB_TARGET_OPENFHEPKE_OPENFHETRANSLATEREGISTRATION_H_
#define LIB_TARGET_OPENFHEPKE_OPENFHETRANSLATEREGISTRATION_H_

namespace mlir {
namespace heir {
namespace openfhe {

void registerTranslateOptions();

void registerToOpenFhePkeTranslation();

void registerToOpenFhePkeHeaderTranslation();

void registerToOpenFhePkePybindTranslation();

void registerToOpenFhePkeDebugHeaderTranslation();

void registerToOpenFhePkeDebugTranslation();

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_TARGET_OPENFHEPKE_OPENFHETRANSLATEREGISTRATION_H_
