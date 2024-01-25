// RUN: not heir-opt --bgv-to-openfhe --split-input-file %s 2>&1

#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>
#rings2 = #bgv.rings<#ring1, #ring2, #ring2>
!ct = !bgv.ciphertext<rings=#rings, dim=2, level=1>

func.func @test_relin_to_basis_error(%x: !bgv.ciphertext<rings=#rings, dim=4, level=1>) {
  // expected-error@+1 {{toBasis must be [0, 1], got [0, 2]}}
  %relin_error = bgv.relinearize(%x) { from_basis = array<i32: 0, 1, 2, 3>, to_basis = array<i32: 0, 2> }: (!bgv.ciphertext<rings=#rings, dim=4, level=1>) -> !ct
  return
}

// -----
#my_poly = #polynomial.polynomial<1 + x**1024>
#ring1 = #polynomial.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #polynomial.ring<cmod=33538049, ideal=#my_poly>
#rings2 = #bgv.rings<#ring1, #ring2, #ring2>
func.func @test_modswitch_level_error(%x: !bgv.ciphertext<rings=#rings2, dim=2, level=2>) {
  // expected-error@+1 {{fromLevel must be toLevel + 1, got fromLevel: 2 and toLevel: 0}}
  %relin_error = bgv.modulus_switch(%x) {
    from_level = 2, to_level = 0
    }: (!bgv.ciphertext<rings=#rings2, dim=2, level=2>) -> !bgv.ciphertext<rings=#rings2, dim=2, level=0>
  return
}
