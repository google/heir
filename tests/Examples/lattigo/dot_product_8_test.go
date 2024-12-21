package dot_product_8

import (
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

func TestBinops(t *testing.T) {
	var err error
	var params bgv.Parameters

	// 128-bit secure parameters enabling depth-7 circuits.
	// LogN:14, LogQP: 431.
	if params, err = bgv.NewParametersFromLiteral(
		bgv.ParametersLiteral{
			LogN:             14,                                    // log2(ring degree)
			LogQ:             []int{55, 45, 45, 45, 45, 45, 45, 45}, // log2(primes Q) (ciphertext modulus)
			LogP:             []int{61},                             // log2(primes P) (auxiliary modulus)
			PlaintextModulus: 0x10001,                               // log2(scale)
		}); err != nil {
		panic(err)
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	ecd := bgv.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, sk)
	dec := rlwe.NewDecryptor(params, sk)
	relinKeys := kgen.GenRelinearizationKeyNew(sk)
	// 5^7 % (2^14) = 12589
	galKey12589 := kgen.GenGaloisKeyNew(12589, sk)
	// 5^4 = 625
	galKey625 := kgen.GenGaloisKeyNew(625, sk)
	// 5^2 = 25
	galKey25 := kgen.GenGaloisKeyNew(25, sk)
	// 5^1 = 5
	galKey5 := kgen.GenGaloisKeyNew(5, sk)
	evalKeys := rlwe.NewMemEvaluationKeySet(relinKeys, galKey5, galKey25, galKey625, galKey12589)
	evaluator := bgv.NewEvaluator(params, evalKeys /*scaleInvariant=*/, false)

	// Vector of plaintext values
	arg0 := []int16{1, 2, 3, 4, 5, 6, 7, 8}
	arg1 := []int16{2, 3, 4, 5, 6, 7, 8, 9}

	expected := int16(240)

	ct0, ct1 := dot_product__encrypt(evaluator, params, ecd, enc, arg0, arg1)

	resultCt := dot_product(evaluator, params, ecd, ct0, ct1)

	result := dot_product__decrypt(evaluator, params, ecd, dec, resultCt)

	if result != expected {
		t.Errorf("Decryption error %d != %d", result, expected)
	}
}
