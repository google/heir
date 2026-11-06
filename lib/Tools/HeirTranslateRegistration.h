#ifndef LIB_TOOLS_HEIRTRANSLATEREGISTRATION_H_
#define LIB_TOOLS_HEIRTRANSLATEREGISTRATION_H_

namespace mlir {
namespace heir {

// Registers all translations (and their command-line options) that
// heir-translate supports with the global translation registry. Must be called
// at most once per process.
void registerAllHEIRTranslations();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TOOLS_HEIRTRANSLATEREGISTRATION_H_
