package chebyshev

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"testing"
)

func TestChebyshev(t *testing.T) {
	num_slots := 32768
	input_clear := make([]float64, num_slots)
	for i := range input_clear {
		input_clear[i] = 2.0
	}

	expected_clear := make([]float64, num_slots)
	// polynomial computes f(x) = x^3
	for i := range expected_clear {
		expected_clear[i] = 8.0
	}

	// These parameters should match the mlir file, though due to the
	// weird nature of this test, this is the source of truth for what
	// is used, not the mlir file.
	param, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,
		Q:               []uint64{36028797019488257, 1099512938497, 1099510054913, 1099507695617, 1099515691009, 1099516870657, 1099506515969, 1099504549889, 1099503894529, 1099503370241, 1099502714881},
		P:               []uint64{2305843009211596801, 2305843009210023937, 2305843009208713217},
		LogDefaultScale: 40,
	})
	if err != nil {
		panic(err)
	}

	encoder := ckks.NewEncoder(param)
	kgen := rlwe.NewKeyGenerator(param)
	sk, pk := kgen.GenKeyPairNew()
	encryptor := rlwe.NewEncryptor(param, pk)
	decryptor := rlwe.NewDecryptor(param, sk)
	rk := kgen.GenRelinearizationKeyNew(sk)
	ekset := rlwe.NewMemEvaluationKeySet(rk)
	evaluator := ckks.NewEvaluator(param, ekset)

	pt := ckks.NewPlaintext(param, param.MaxLevel())
	encoder.Encode(input_clear, pt)
	ct_input, err25 := encryptor.EncryptNew(pt)
	if err25 != nil {
		panic(err25)
	}

	result_ct := chebyshev(evaluator, param, encoder, ct_input)
	result_pt := decryptor.DecryptNew(result_ct)
	result_float64 := make([]float64, num_slots)
	encoder.Decode(result_pt, result_float64)

	epsilon := 1e-05
	for i := 0; i < num_slots; i++ {
		diff := math.Abs(result_float64[i] - expected_clear[i])
		if diff > epsilon {
			t.Errorf("Mismatch at index %d: got %f, expected %f (diff: %e)",
				i, result_float64[i], expected_clear[i], diff)

			// Fail fast to avoid spamming 4096 errors
			if i > 10 {
				t.Fatal("Too many errors, stopping verification.")
			}
		}
	}
}
