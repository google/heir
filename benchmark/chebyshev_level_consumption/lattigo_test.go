package main

import (
	"math/bits"
	"testing"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/polynomial"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/bignum"
)

func TestLattigoFormulaMatch(t *testing.T) {
	var err error
	var params ckks.Parameters

	if params, err = ckks.NewParametersFromLiteral(
		ckks.ParametersLiteral{
			LogN:            14,                                                // ring degree 16384
			LogQ:            []int{50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, // 11 primes, depth 10
			LogP:            []int{50},                                         // auxiliary modulus
			LogDefaultScale: 40,
			RingType:        ring.Standard,
		}); err != nil {
		t.Fatalf("failed to create parameters: %v", err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	ecd := ckks.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	eval := ckks.NewEvaluator(params, evk)

	polyEval := polynomial.NewEvaluator(params, eval)

	for d := 1; d <= 40; d++ {
		pt := ckks.NewPlaintext(params, params.MaxLevel())
		values := make([]float64, pt.Slots())
		for i := range values {
			values[i] = 1.0
		}
		if err = ecd.Encode(values, pt); err != nil {
			t.Fatalf("failed to encode: %v", err)
		}
		var ct *rlwe.Ciphertext
		if ct, err = enc.EncryptNew(pt); err != nil {
			t.Fatalf("failed to encrypt: %v", err)
		}

		coeffs := make([]float64, d+1)
		for i := range coeffs {
			coeffs[i] = 1.0
		}
		interval := [2]float64{-1.0, 1.0}
		bignumPoly := bignum.NewPolynomial(bignum.Chebyshev, coeffs, interval)
		poly := polynomial.NewPolynomial(bignumPoly)

		initialLevel := ct.Level()

		var res *rlwe.Ciphertext
		if res, err = polyEval.Evaluate(ct, poly, params.DefaultScale()); err != nil {
			t.Fatalf("failed to evaluate: %v", err)
		}

		levelsConsumed := initialLevel - res.Level()
		expected := bits.Len64(uint64(d))

		if levelsConsumed != expected {
			t.Errorf("Degree: %d: levels consumed %d, expected %d", d, levelsConsumed, expected)
		}
	}
}
