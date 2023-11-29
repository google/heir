// RUN: heir-opt --verify-diagnostics --split-input-file %s

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>

func.func @test_input_dimension_error(%input: !bgv.ciphertext<rings=#rings, dim=3>) {
  // expected-error@+1 {{x.dim == 2 does not hold}}
  %out = bgv.rotate (%input) {offset = 4} : (!bgv.ciphertext<rings=#rings, dim=3>) -> !bgv.ciphertext<rings=#rings>

  return
}
