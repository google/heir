#ifndef HEIR_LIB_APPROXIMATION_SIGN_H_
#define HEIR_LIB_APPROXIMATION_SIGN_H_

//
// Polynomial approximation for sign
//
//   Poly(-260.03867215588*x**11 + 746.781707684981*x**9 - 797.090149675776*x**7
//   + 388.964712077092*x**5 - 86.6415008377027*x**3 + 8.82341343192733*x, x,
//   domain='RR')
//
// Generated via
//
//   python scripts/convert_lolremez.py \
//     --name='sign' \
//     --poly_str='((((-2.6003867215587949e+2*x+7.4678170768498101e+2)*x-7.9709014967577619e+2)*x+3.8896471207709223e+2)*x-8.6641500837702735e+1)*x+8.8234134319273287'
//     \
//     --use_odd_trick='True'
//
// using output from
//
//   lolremez -d 5 -r '0.01:1' '1/sqrt(x)' '1/sqrt(x)'
//
static constexpr double SIGN_APPROX_COEFFICIENTS[12] = {
    0.0000000000000000,   8.8234134319273300,    0.0000000000000000,
    -86.6415008377027000, 0.0000000000000000,    388.9647120770920000,
    0.0000000000000000,   -797.0901496757760000, 0.0000000000000000,
    746.7817076849810000, 0.0000000000000000,    -260.0386721558800000};
static constexpr int SIGN_APPROX_LEN = 12;

#endif  // HEIR_LIB_APPROXIMATION_SIGN_H_
