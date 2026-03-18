package lineartransform

import (
	"math"
	"testing"

	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func TestLinearTransform(t *testing.T) {
	evaluator, params, encoder, encryptor, decryptor := linear_transform__configure()
	numSlots := params.MaxSlots()

	inputClear := make([]float64, numSlots)
	for i := range inputClear {
		inputClear[i] = 1.0
	}

	// Matrix of weights: two nonzero diagonals on an otherwise all-zero matrix.
	//
	// diagonal 0: [0, 1, 2, ..., 4095]
	// diagonal 1: [4096, 4097, ..., 8191]
	//
	// With an all-1s input vector, expected output = diagonal0 + diagonal1:
	// [4096+0, 4097+1, ..., 8191+4095]
	diagonals := 2
	matrix := make([]float64, diagonals*numSlots)
	value := 0.0
	for r := 0; r < diagonals; r++ {
		for c := 0; c < numSlots; c++ {
			matrix[r*numSlots+c] = value
			value++
		}
	}

	expectedClear := make([]float64, numSlots)
	for i := range expectedClear {
		expectedClear[i] = float64(numSlots + 2*i)
	}

	pt := ckks.NewPlaintext(params, params.MaxLevel())
	pt.Scale = params.DefaultScale()
	if err := encoder.Encode(inputClear, pt); err != nil {
		t.Fatal(err)
	}
	ct, err := encryptor.EncryptNew(pt)
	if err != nil {
		t.Fatal(err)
	}

	resultCt := linear_transform(evaluator, params, encoder, ct, matrix)
	resultPt := decryptor.DecryptNew(resultCt)
	resultFloat64 := make([]float64, numSlots)
	if err := encoder.Decode(resultPt, resultFloat64); err != nil {
		t.Fatal(err)
	}

	// Scale 26 is not very precise; epsilon of 1.5 is sufficient here.
	epsilon := 1.5
	for i := 0; i < numSlots; i++ {
		diff := math.Abs(resultFloat64[i] - expectedClear[i])
		if diff > epsilon {
			t.Errorf("Mismatch at index %d: got %f, expected %f (diff: %e)",
				i, resultFloat64[i], expectedClear[i], diff)
			if i > 10 {
				t.Fatal("Too many errors, stopping verification.")
			}
		}
	}
}
