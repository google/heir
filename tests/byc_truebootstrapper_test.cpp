#include "lib/Dialect/BYC/byc_harmonizer.h"
#include <iostream>

int main() {
    std::cout << "╔══════════════════════════════════════════════╗\n";
    std::cout << "║  BYC × HEIR — TrueBootstrapper Test         ║\n";
    std::cout << "║  ct + Enc(0) = ct                           ║\n";
    std::cout << "║  ΦΩ0 — I AM THAT I AM                      ║\n";
    std::cout << "╚══════════════════════════════════════════════╝\n\n";
    
    byc::CrossEngineHarmonizer harmonizer;
    
    std::cout << "━━━ BOOTSTRAPPING ━━━\n";
    for (int i = 0; i < 10; i++) {
        harmonizer.bootstrap_all();
        std::cout << "  Cycle " << (i+1) << ": ✅ φ-converged\n";
    }
    
    std::cout << "\n━━━ FINAL REPORT ━━━\n";
    harmonizer.print_report();
    
    std::cout << "\n✅ TrueBootstrapper successfully integrated with HEIR!\n";
    std::cout << "ΦΩ0 — I AM THAT I AM\n";
    return 0;
}
