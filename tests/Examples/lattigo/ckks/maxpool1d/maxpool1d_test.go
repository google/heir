package maxpool1d

import (
	"math"
	"testing"
)

func TestMaxPool1d(t *testing.T) {
	evaluator, params, ecd, enc, dec := Maxpool1d__configure()

	// Input: 1x3x16, output: 1x3x8 with kernel_size=2, stride=2.
	const C = 3
	const W = 16
	const OW = 8
	// Inputs in [-0.5, 0.5], so that the difference of each pair lies in the
	// default approximation domain [-1, 1] of the max.
	arg0 := make([]float32, C*W)
	for i := range arg0 {
		arg0[i] = float32(0.5 * math.Sin(float64(i)*1.3))
	}

	expected := make([]float32, C*OW)
	for c := 0; c < C; c++ {
		for i := 0; i < OW; i++ {
			a := arg0[c*W+2*i]
			b := arg0[c*W+2*i+1]
			expected[c*OW+i] = float32(math.Max(float64(a), float64(b)))
		}
	}

	ct0 := Maxpool1d__encrypt__arg0(evaluator, params, ecd, enc, arg0)
	resultCt := Maxpool1d(evaluator, params, ecd, ct0)
	result := Maxpool1d__decrypt__result0(evaluator, params, ecd, dec, resultCt)

	// The max is approximated by a low-degree polynomial, whose error is
	// largest (about 0.04) when the two elements are close.
	errorThreshold := float64(0.1)
	maxError := float64(0)
	for i := range expected {
		maxError = math.Max(maxError, math.Abs(float64(result[i]-expected[i])))
		if math.Abs(float64(result[i]-expected[i])) > errorThreshold {
			t.Errorf("Decryption error at index %d: %.4f != %.4f", i, result[i], expected[i])
		}
	}
	t.Logf("Max error: %.4f", maxError)
}
