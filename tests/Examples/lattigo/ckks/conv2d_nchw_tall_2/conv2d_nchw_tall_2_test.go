package conv2dnchwtall2

import (
	"math"
	"testing"
)

func TestConv2D(t *testing.T) {
	evaluator, params, ecd, enc, dec := conv2d_nchw__configure()

	cols := 64 // input elements
	arg0 := make([]float32, cols)
	for i := 0; i < cols; i++ {
		arg0[i] = float32(i)
	}

	expectedSingleChannel := []float32{18, 26, 34, 42, 82, 90, 98, 106, 146, 154, 162, 170, 210, 218, 226, 234}
	expected := make([]float32, 128)
	for c := 0; c < 8; c++ {
		for i := 0; i < 16; i++ {
			expected[c*16+i] = expectedSingleChannel[i]
		}
	}

	ct0 := conv2d_nchw__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := conv2d_nchw(evaluator, params, ecd, ct0)
	result := conv2d_nchw__decrypt__result0(evaluator, params, ecd, dec, resultCt)
	errorThreshold := float64(0.5)
	for i := 0; i < 128; i++ {
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.2f != %.2f", i, result[i], expected[i])
		}
	}
}
