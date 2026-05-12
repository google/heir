package conv1dncw

import (
	"math"
	"testing"
)

func TestConv1D(t *testing.T) {
	evaluator, params, ecd, enc, dec := conv1d_ncw__configure()

	cols := 4 // input elements
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = float32(i)
	}

	expectedSingleChannel := []float32{1, 5}
	expected := make([]float32, 16)
	for c := 0; c < 8; c++ {
		for i := 0; i < 2; i++ {
			expected[c*2+i] = expectedSingleChannel[i]
		}
	}

	ct0 := conv1d_ncw__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := conv1d_ncw(evaluator, params, ecd, ct0)
	result := conv1d_ncw__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.5)
	for i := 0; i < 16; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected[i])
		}
	}
}
