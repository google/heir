package chebyshev

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"math"
	"testing"
)

func TestChebyshev(t *testing.T) {
	numSlots := 32768
	inputClear := make([]float64, numSlots)
	for i := range inputClear {
		inputClear[i] = 2.0
	}

	expectedClear := make([]float64, numSlots)
	// polynomial computes f(x) = x^3
	for i := range expectedClear {
		expectedClear[i] = 8.0
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
	encoder.Encode(inputClear, pt)
	ctInput, err25 := encryptor.EncryptNew(pt)
	if err25 != nil {
		panic(err25)
	}

	resultCt := chebyshev(evaluator, param, encoder, ctInput)
	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, numSlots)
	encoder.Decode(resultPt, resultFloat64)

	epsilon := 1e-05
	for i := 0; i < numSlots; i++ {
		diff := math.Abs(resultFloat64[i] - expectedClear[i])
		if diff > epsilon {
			t.Errorf("Mismatch at index %d: got %f, expected %f (diff: %e)",
				i, resultFloat64[i], expectedClear[i], diff)

			// Fail fast to avoid spamming 4096 errors
			if i > 10 {
				t.Fatal("Too many errors, stopping verification.")
			}
		}
	}
}
